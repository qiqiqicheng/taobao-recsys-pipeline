"""
List-wise retrieval training (CrossEntropy with in-batch negatives).

This module intentionally supports only the list-wise formulation (equivalent to the old mode=2).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Protocol, cast

import lightning as L
import torch
import torch.nn as nn

from taobao_recsys_pipeline.losses.losses import RegularizationLoss
from taobao_recsys_pipeline.metrics.match_metrics import MatchMetrics
from taobao_recsys_pipeline.utils.pylogger import RankedLogger

log = RankedLogger(__name__)


class SupportsTwoTowers(Protocol):
    def user_tower(self, x: dict[str, torch.Tensor]) -> torch.Tensor: ...

    def item_tower(self, x: dict[str, torch.Tensor]) -> torch.Tensor: ...

    def set_mode(self, mode: Literal["user", "item"]) -> None: ...


def _inbatch_negative_sampling(
    scores: torch.Tensor,
    neg_ratio: int | None,
    hard_negative: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Return indices of selected negatives for each sample in the batch.

    Parameters
    ----------
    scores:     [B, B] inner-product matrix (diagonal = positive pairs).
    neg_ratio:  number of negatives per sample; None → use all off-diagonal.
    hard_negative: if True pick top-(neg_ratio) by score instead of random.

    Returns
    -------
    Tensor[B, neg_ratio]  index tensor into the B items dimension.
    """
    B = scores.size(0)
    device = scores.device

    # Mask out diagonal (positive)
    mask = ~torch.eye(B, dtype=torch.bool, device=device)  # [B, B]
    neg_scores = scores.masked_fill(~mask, float("-inf"))  # [B, B]

    k = neg_ratio if neg_ratio is not None else B - 1

    if hard_negative:
        _, neg_indices = torch.topk(neg_scores, k, dim=1)
    else:
        # Sample without replacement from valid negatives
        probs = mask.float()  # uniform over valid positions
        neg_indices = torch.multinomial(probs, num_samples=k, replacement=False, generator=generator)  # [B, k], dim=1

    return neg_indices


def _gather_inbatch_logits(
    scores: torch.Tensor,
    neg_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather [B, 1+k] logit matrix: column 0 = positive, columns 1..k = negatives.

    Parameters
    ----------
    scores:       [B, B]
    neg_indices:  [B, k]

    Returns
    -------
    Tensor[B, 1+k]
    """
    B = scores.size(0)
    device = scores.device
    pos_logits = scores[torch.arange(B, device=device), torch.arange(B, device=device)].unsqueeze(1)  # [B, 1]
    neg_logits = scores.gather(1, neg_indices)  # [B, k]
    return torch.cat([pos_logits, neg_logits], dim=1)  # [B, 1+k]


class MatchModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        in_batch_neg: bool = True,
        in_batch_neg_ratio: int | None = None,
        hard_negative: bool = False,
        sampler_seed: int | None = None,
        temperature: float = 1.0,
        optimizer: Callable = torch.optim.AdamW,
        scheduler: Callable | None = None,
        reg_loss: RegularizationLoss | None = None,
        val_at_k_list: list[int] = [10, 50],
    ) -> None:
        super().__init__()
        self.model = model
        self.tower_model = cast(SupportsTwoTowers, model)
        self.in_batch_neg = in_batch_neg
        self.in_batch_neg_ratio = in_batch_neg_ratio
        self.hard_negative = hard_negative
        self.temperature = temperature
        self.val_at_k_list = val_at_k_list

        self._optimizer = optimizer
        self._scheduler = scheduler

        # Regularization
        self.reg_loss_fn = reg_loss if reg_loss is not None else RegularizationLoss()

        # In-batch sampler RNG
        self._sampler_generator: torch.Generator | None = None
        if sampler_seed is not None:
            # Generator will be moved to the correct device in on_fit_start
            self._sampler_seed = sampler_seed
        else:
            self._sampler_seed = None

        # Validate model contract when in_batch_neg is requested
        if in_batch_neg and not (hasattr(model, "user_tower") and hasattr(model, "item_tower")):
            raise ValueError(
                f"{type(model).__name__} does not expose user_tower/item_tower. "
                "in_batch_neg requires a two-tower model."
            )

        # Validation metrics (only meaningful for two-tower in-batch neg models)
        if in_batch_neg:
            self.val_metrics = MatchMetrics(k=max(val_at_k_list), at_k_list=val_at_k_list)

        # Build criterion
        self.criterion = self._build_criterion()

    def on_fit_start(self) -> None:
        if self._sampler_seed is not None:
            self._sampler_generator = torch.Generator(device=self.device)
            self._sampler_generator.manual_seed(self._sampler_seed)

    def forward(self, x: dict[str, torch.Tensor]) -> Any:
        return self.model(x)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        _ = batch_idx
        loss = self._compute_loss(batch, stage="train")
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        _ = batch_idx
        self._compute_loss(batch, stage="val")

    def on_validation_epoch_end(self) -> None:
        if self.in_batch_neg:
            metrics = self.val_metrics.compute()
            for name, value in metrics.items():
                self.log(f"val/{name}", value, prog_bar=True)
            self.val_metrics.reset()

    def _compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        stage: str,
    ) -> torch.Tensor:
        if not self.in_batch_neg:
            raise ValueError

        loss = self._inbatch_neg_step(batch, stage=stage)

        # Regularization
        reg_loss = self.reg_loss_fn(self.model)
        total_loss = loss + reg_loss

        self.log(f"{stage}/loss", total_loss, prog_bar=(stage == "train"), on_step=(stage == "train"), on_epoch=True)

        return total_loss

    def _inbatch_neg_step(
        self,
        batch: dict[str, torch.Tensor],
        stage: str,
    ) -> torch.Tensor:
        """In-batch negative sampling step for two-tower models."""
        user_emb = self.tower_model.user_tower(batch)  # [B, D]
        item_emb = self.tower_model.item_tower(batch)  # [B, D]

        # Squeeze extra dims that some models keep for inference convenience
        if user_emb.dim() > 2:
            user_emb = user_emb.squeeze(1)
        if item_emb.dim() > 2:
            item_emb = item_emb.squeeze(1)

        # scores[i, j] = sim(user_i, item_j); diagonal is the positive pairs
        scores = torch.matmul(user_emb, item_emb.t()) / self.temperature  # [B, B]
        B = scores.size(0)

        # Optionally subsample negatives per sample
        if self.in_batch_neg_ratio is not None:
            neg_indices = _inbatch_negative_sampling(
                scores, self.in_batch_neg_ratio, self.hard_negative, self._sampler_generator
            )
            logits = _gather_inbatch_logits(scores, neg_indices)  # [B, 1+k]
            targets = torch.zeros(B, dtype=torch.long, device=self.device)  # positive is always column 0
        else:
            logits = scores  # [B, B] (pos on diagonal)
            targets = torch.arange(B, device=self.device)

        loss = self.criterion(logits, targets)

        # Accumulate MatchMetrics during validation
        if stage == "val":
            k = min(max(self.val_at_k_list), B - 1)
            top_k_ids = scores.topk(k, dim=1).indices  # [B, k]
            target_ids = torch.arange(B, device=self.device)  # [B]
            self.val_metrics.update(top_k_ids, target_ids)

        return loss

    def _pairwise_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        _ = batch
        raise NotImplementedError

    def _pointwise_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        _ = batch
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = self._optimizer(self.model.parameters())
        if self._scheduler is None:
            return optimizer
        scheduler = self._scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            },
        }

    @torch.no_grad()
    def inference_embedding(
        self,
        dataloader: torch.utils.data.DataLoader,
        mode: Literal["user", "item"],
        ckpt_path: str | None = None,
    ) -> torch.Tensor:
        """Generate user or item embeddings for the entire dataloader.

        Parameters
        ----------
        dataloader:
            Yields dicts of tensors (no labels required).
        mode:
            ``"user"`` → calls ``model.user_tower``; ``"item"`` → ``model.item_tower``.
        ckpt_path:
            Optional path to a ``.ckpt`` Lightning checkpoint to load before inference.

        Returns
        -------
        Tensor[N, D]
        """
        if ckpt_path is not None:
            state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            # Lightning checkpoints store model weights under "state_dict"
            prefix = "model."
            model_state = {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state["state_dict"].items()}
            self.model.load_state_dict(model_state)
            log.info(f"Loaded checkpoint from {ckpt_path}")

        if not (hasattr(self.model, "user_tower") and hasattr(self.model, "item_tower")):
            raise ValueError(f"{type(self.model).__name__} does not support set_mode() / tower inference.")

        if hasattr(self.tower_model, "set_mode"):
            self.tower_model.set_mode(mode)

        self.model.eval()
        tower_fn = self.tower_model.user_tower if mode == "user" else self.tower_model.item_tower

        all_embeddings: list[torch.Tensor] = []
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            emb = tower_fn(batch)
            if emb.dim() > 2:
                emb = emb.squeeze(1)
            all_embeddings.append(emb)

        return torch.cat(all_embeddings, dim=0)  # [N, D]

    def _build_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

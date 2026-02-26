"""
  0 – point-wise  : BCELoss (w/ explicit labels) or CrossEntropyLoss (w/ in-batch negatives)
  1 – pair-wise   : BPRLoss(pos_score, neg_score)
  2 – list-wise   : CrossEntropyLoss  (default for retrieval, works best with in-batch negatives)

Model contract
--------------
Two-tower models (YoutubeDNN, SASRec, …) should expose:
    user_tower(x: dict) -> Tensor[B, D]
    item_tower(x: dict) -> Tensor[B, D]
    set_mode(mode: Literal["user", "item"])   # used at inference time

Single-tower / interaction models (ItemCF proxy, MF, …) only need:
    forward(x: dict) -> Tensor[B]             # point-wise probability / score
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import lightning as L
import torch
import torch.nn as nn

from taobao_recsys_pipeline.losses.losses import BPRLoss, RegularizationLoss
from taobao_recsys_pipeline.metrics.match_metrics import MatchMetrics
from taobao_recsys_pipeline.utils.pylogger import RankedLogger

log = RankedLogger(__name__)


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
        mode: Literal[0, 1, 2] = 2,
        in_batch_neg: bool = True,
        in_batch_neg_ratio: int | None = None,
        hard_negative: bool = False,
        sampler_seed: int | None = None,
        temperature: float = 1.0,
        optimizer: Callable = torch.optim.Adam,
        scheduler: Callable | None = None,
        reg_loss: RegularizationLoss | None = None,
        val_at_k_list: list[int] = [10, 50],
    ) -> None:
        super().__init__()
        self.model = model
        self.mode = mode
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
        loss = self._compute_loss(batch, stage="train")
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
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
        if self.in_batch_neg:
            loss = self._inbatch_neg_step(batch, stage=stage)
        elif self.mode == 1:  # pair-wise
            loss = self._pairwise_step(batch)
        else:  # point-wise
            loss = self._pointwise_step(batch)

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
        user_emb = self.model.user_tower(batch)  # [B, D]
        item_emb = self.model.item_tower(batch)  # [B, D]

        # Squeeze extra dims that some models keep for inference convenience
        if user_emb.dim() > 2:
            user_emb = user_emb.squeeze(1)
        if item_emb.dim() > 2:
            item_emb = item_emb.squeeze(1)

        scores = torch.matmul(user_emb, item_emb.t()) / self.temperature  # [B, B]
        B = scores.size(0)

        if self.mode == 1:  # pair-wise BPR via in-batch negatives
            neg_indices = _inbatch_negative_sampling(
                scores, self.in_batch_neg_ratio, self.hard_negative, self._sampler_generator
            )  # [B, k]
            logits = _gather_inbatch_logits(scores, neg_indices)  # [B, 1+k]
            loss = self.criterion(logits[:, 0], logits[:, 1:].mean(dim=1))  # BPR: mean of negatives
        else:
            # Default: list-wise / point-wise via full-batch CrossEntropy
            # Optionally subsample negatives per sample
            if self.in_batch_neg_ratio is not None:
                neg_indices = _inbatch_negative_sampling(
                    scores, self.in_batch_neg_ratio, self.hard_negative, self._sampler_generator
                )
                logits = _gather_inbatch_logits(scores, neg_indices)  # [B, 1+k]
            else:
                logits = scores  # [B, B]  (pos on diagonal)

            targets = torch.arange(B, device=self.device)
            loss = self.criterion(logits, targets)

        # Accumulate MatchMetrics during validation
        if stage == "val":
            k = max(self.val_at_k_list)
            top_k_ids = scores.topk(k, dim=1).indices  # [B, k]
            target_ids = torch.arange(B, device=self.device)  # [B]
            self.val_metrics.update(top_k_ids, target_ids)

        return loss

    def _pairwise_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Pair-wise BPR step.  Model must return (pos_score, neg_score)."""
        pos_score, neg_score = self.model(batch)
        return self.criterion(pos_score, neg_score)

    def _pointwise_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Point-wise step.  Batch must contain a 'label' key."""
        if "label" not in batch:
            raise KeyError(
                "Point-wise mode (mode=0, in_batch_neg=False) expects a 'label' key in the batch. "
                "Add it in the Dataset or switch to in_batch_neg=True."
            )
        y = batch["label"].float()
        y_pred = self.model(batch).squeeze(-1)
        return self.criterion(y_pred, y)

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

        if hasattr(self.model, "set_mode"):
            self.model.set_mode(mode)

        self.model.eval()
        tower_fn = self.model.user_tower if mode == "user" else self.model.item_tower

        all_embeddings: list[torch.Tensor] = []
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            emb = tower_fn(batch)
            if emb.dim() > 2:
                emb = emb.squeeze(1)
            all_embeddings.append(emb)

        return torch.cat(all_embeddings, dim=0)  # [N, D]

    def _build_criterion(self) -> nn.Module:
        if self.mode == 0:
            return nn.CrossEntropyLoss() if self.in_batch_neg else nn.BCELoss()
        elif self.mode == 1:
            return BPRLoss()
        elif self.mode == 2:
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"mode must be 0/1/2, got {self.mode}")

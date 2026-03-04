from __future__ import annotations

from collections.abc import Callable

import lightning as L
import torch
import torch.nn as nn

from taobao_recsys_pipeline.losses.losses import RegularizationLoss
from taobao_recsys_pipeline.metrics.ctr_metrics import CTRMetrics
from taobao_recsys_pipeline.utils.pylogger import RankedLogger

log = RankedLogger(__name__)


class MissingCandidateFeaturesError(KeyError):
    pass


class InvalidCandidateShapeError(ValueError):
    pass


class CTRModule(L.LightningModule):
    """List-wise CTR module over sampled candidates.

    The datamodule provides one positive + (K-1) negatives in
    ``candidate_item_id`` / ``candidate_cat_id`` per sample. This module builds
    candidate-wise labels inside ``_compute_loss`` and optimizes list-wise
    CrossEntropy where the positive class index is 0.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Callable = torch.optim.AdamW,
        scheduler: Callable | None = None,
        reg_loss: RegularizationLoss | None = None,
        val_at_k_list: list[int] | None = None,
    ):
        super().__init__()
        self.model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self.reg_loss_fn = reg_loss if reg_loss else RegularizationLoss()
        self.val_at_k_list = val_at_k_list if val_at_k_list is not None else [1, 5, 10]

        # DIN forward returns probabilities in [0, 1]; convert to logits before CE.
        self.criterion = torch.nn.CrossEntropyLoss()

        # Flattened binary diagnostics (AUC/acc/precision/recall/f1/logloss)
        self.val_metrics = CTRMetrics()

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(x)

    def _build_listwise_scores_and_labels(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build [B, K] candidate scores and labels from a batch.

        Returns
        -------
        probs:
            [B, K] predicted click probabilities for each candidate.
        logits:
            [B, K] logits derived from probs for CrossEntropy.
        ce_targets:
            [B] class indices for CE (always 0 because positive is prepended).
        binary_labels:
            [B, K] one-hot labels for metric diagnostics.
        """
        if "candidate_item_id" not in batch or "candidate_cat_id" not in batch:
            raise MissingCandidateFeaturesError

        cand_item = batch["candidate_item_id"].long()  # [B, K]
        cand_cat = batch["candidate_cat_id"].long()  # [B, K]
        if cand_item.dim() != 2:
            raise InvalidCandidateShapeError

        B, K = cand_item.shape

        # Build one-hot list-wise labels in-module: positive is always column 0.
        binary_labels = torch.zeros(B, K, dtype=torch.float32, device=self.device)
        binary_labels[:, 0] = 1.0
        ce_targets = torch.zeros(B, dtype=torch.long, device=self.device)

        # Expand each sample to K candidate rows, then evaluate DIN in one forward.
        flat_batch: dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if key in {"candidate_item_id", "candidate_cat_id", "label"}:
                continue
            if not torch.is_tensor(value):
                continue
            flat_batch[key] = value.repeat_interleave(K, dim=0)

        flat_batch["target_item_id"] = cand_item.reshape(B * K)
        flat_batch["target_cat_id"] = cand_cat.reshape(B * K)

        probs = self(flat_batch).float().view(B, K)
        probs = probs.clamp(1e-6, 1.0 - 1e-6)
        logits = torch.logit(probs)

        return probs, logits, ce_targets, binary_labels

    @staticmethod
    def _compute_batch_ranking_metrics(
        logits: torch.Tensor,
        targets: torch.Tensor,
        at_k_list: list[int],
    ) -> dict[str, torch.Tensor]:
        """Compute HR@k and MRR for one batch."""
        _, K = logits.shape
        ranked_indices = logits.argsort(dim=1, descending=True)  # [B, K]

        pos = targets.view(-1, 1)
        hit_pos = (ranked_indices == pos).nonzero(as_tuple=False)
        # each row has exactly one match
        ranks = hit_pos[:, 1].float() + 1.0

        out: dict[str, torch.Tensor] = {"mrr": (1.0 / ranks).mean()}
        for at_k in at_k_list:
            k = min(at_k, K)
            out[f"hr@{at_k}"] = (ranks <= float(k)).float().mean()
        return out

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        _ = batch_idx
        loss = self._compute_loss(batch, stage="train")
        return loss

    def _compute_loss(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        probs, logits, ce_targets, binary_labels = self._build_listwise_scores_and_labels(batch)

        ce_loss = self.criterion(logits, ce_targets)
        reg_loss = self.reg_loss_fn(self.model)
        total_loss = ce_loss + reg_loss

        self.log(
            f"{stage}/loss",
            total_loss,
            prog_bar=(stage == "train"),
            on_step=(stage == "train"),
            on_epoch=True,
        )

        if stage == "val":
            # flattened binary diagnostics
            self.val_metrics.update(probs.view(-1), binary_labels.view(-1))

            # list-wise ranking diagnostics
            ranking = self._compute_batch_ranking_metrics(logits, ce_targets, self.val_at_k_list)
            for name, value in ranking.items():
                self.log(
                    f"val/{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=(name == "mrr"),
                    batch_size=logits.size(0),
                )

        return total_loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        _ = batch_idx
        _ = self._compute_loss(batch, stage="val")

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        for name, value in metrics.items():
            self.log(f"val/{name}", value, prog_bar=False)
        self.val_metrics.reset()

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

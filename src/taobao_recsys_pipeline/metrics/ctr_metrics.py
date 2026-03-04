from __future__ import annotations

from typing import cast

import torch
import torchmetrics
import torchmetrics.utilities


class CTRMetrics(torchmetrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor, **kwargs):
        _ = kwargs
        # store detached tensors to avoid holding graph
        cast(list[torch.Tensor], self.preds).append(preds.detach())
        cast(list[torch.Tensor], self.targets).append(targets.detach())

    def compute(self):
        preds = torchmetrics.utilities.dim_zero_cat(cast(list[torch.Tensor], self.preds))
        targets = torchmetrics.utilities.dim_zero_cat(cast(list[torch.Tensor], self.targets))

        preds = preds.float().view(-1)
        targets = targets.float().view(-1)

        # For metrics that expect probabilities in [0,1]
        probs = preds.clamp(0.0, 1.0)
        pred_label = (probs >= 0.5).to(torch.int32)
        true_label = targets.to(torch.int32)

        out: dict[str, torch.Tensor] = {}

        # torchmetrics.functional.* return type annotations may include None in some stubs;
        # at runtime they return tensors for valid inputs.
        out["auc"] = cast(torch.Tensor, torchmetrics.functional.auroc(probs, true_label, task="binary"))
        out["acc"] = cast(torch.Tensor, torchmetrics.functional.accuracy(pred_label, true_label, task="binary"))
        out["precision"] = cast(torch.Tensor, torchmetrics.functional.precision(pred_label, true_label, task="binary"))
        out["recall"] = cast(torch.Tensor, torchmetrics.functional.recall(pred_label, true_label, task="binary"))
        out["f1"] = cast(torch.Tensor, torchmetrics.functional.f1_score(pred_label, true_label, task="binary"))

        # logloss / BCE
        eps = 1e-8
        out["logloss"] = -(targets * torch.log(probs + eps) + (1.0 - targets) * torch.log(1.0 - probs + eps)).mean()
        return out

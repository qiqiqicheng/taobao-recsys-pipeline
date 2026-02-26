import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    """Bayesian Personalised Ranking loss.

    loss = -mean( log σ(pos_score - neg_score) )
    """

    def forward(self, pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
        return -torch.mean(F.logsigmoid(pos_score - neg_score))


class RegularizationLoss(nn.Module):
    """Optional L1/L2 regularisation on embedding and dense parameters."""

    def __init__(
        self,
        embedding_l1: float = 0.0,
        embedding_l2: float = 0.0,
        dense_l1: float = 0.0,
        dense_l2: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding_l1 = embedding_l1
        self.embedding_l2 = embedding_l2
        self.dense_l1 = dense_l1
        self.dense_l2 = dense_l2

    def forward(self, model: nn.Module) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            is_embed = "embed" in name or "embedding" in name
            if is_embed:
                if self.embedding_l1:
                    loss = loss + self.embedding_l1 * param.abs().sum()
                if self.embedding_l2:
                    loss = loss + self.embedding_l2 * (param**2).sum()
            else:
                if self.dense_l1:
                    loss = loss + self.dense_l1 * param.abs().sum()
                if self.dense_l2:
                    loss = loss + self.dense_l2 * (param**2).sum()
        return loss

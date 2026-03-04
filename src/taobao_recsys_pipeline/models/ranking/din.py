from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from taobao_recsys_pipeline.basic.features import BaseFeature
from taobao_recsys_pipeline.basic.layers import MLP, EmbeddingLayer
from taobao_recsys_pipeline.utils.pylogger import RankedLogger

log = RankedLogger(__name__)


class DIN(nn.Module):
    def __init__(
        self,
        session_features: list[BaseFeature],
        history_features: list[BaseFeature],
        target_features: list[BaseFeature],
        mlp: Callable,
        activation_unit: Callable,
    ):
        super().__init__()
        """DIN (Deep Interest Network) for CTR prediction."""
        self.session_features = session_features
        self.history_features = history_features
        self.target_features = target_features
        self.num_history_features = len(history_features)
        self.all_dims = sum([fea.embed_dim for fea in session_features + history_features + target_features])

        self.embedding = EmbeddingLayer(session_features + history_features + target_features)
        self.attention_layers = nn.ModuleList([activation_unit(fea.embed_dim) for fea in self.history_features])
        self.mlp = mlp(self.all_dims, activation="dice")

    def forward(self, x) -> torch.Tensor:
        embed_x_session_features = self.embedding(x, self.session_features)  # [B, session_all_dims, D]
        embed_x_history = self.embedding(x, self.history_features)  # [B, num_history_features, T, D]
        embed_x_target = self.embedding(x, self.target_features)  # [B, num_target_features, D]
        attention_pooling = []

        for i in range(self.num_history_features):
            attention_seq = self.attention_layers[i](embed_x_history[:, i, :, :], embed_x_target[:, i, :])  # [B, D]
            attention_pooling.append(attention_seq.unsqueeze(1))  # [B, 1, D]

        attention_pooling = torch.cat(attention_pooling, dim=1)  # [B, num_history_features, D]
        mlp_in = torch.cat(
            [
                attention_pooling.flatten(start_dim=1),
                embed_x_session_features.flatten(start_dim=1),
                embed_x_target.flatten(start_dim=1),
            ],
            dim=1,
        )  # [B, N]

        y = self.mlp(mlp_in)
        return torch.sigmoid(y.squeeze(1))  # [B,]


class ActivationUnit(nn.Module):
    def __init__(
        self,
        emb_dim,
        dims: list[int] = [36],
        activation="dice",
        use_softmax=False,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.use_softmax = use_softmax
        self.mlp = MLP(input_dim=4 * emb_dim, output_layer=True, dims=dims, activation=activation)

    def forward(self, history: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history (torch.Tensor): [B, T, D]
            target (torch.Tensor): [B, D]

        Returns:
            torch.Tensor: [B, D]
        """
        seq_len = history.size(1)
        target_expanded = target.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, D]
        attn_input = torch.cat(
            [target_expanded, history, target_expanded - history, target_expanded * history], dim=-1
        )  # [B, T, 4D]

        attn_weight = self.mlp(attn_input.view(-1, 4 * self.emb_dim))  # [B*T, 1]
        attn_weight = attn_weight.view(-1, seq_len)  # [B, T]
        if self.use_softmax:
            attn_weight = F.softmax(attn_weight, dim=1)  # [B, T]

        output = (attn_weight.unsqueeze(-1) * history).sum(dim=1)  # [B, D]
        return output

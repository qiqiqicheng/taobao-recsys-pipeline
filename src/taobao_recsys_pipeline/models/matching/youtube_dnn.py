from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F

from taobao_recsys_pipeline.basic.features import BaseFeature
from taobao_recsys_pipeline.basic.layers import EmbeddingLayer
from taobao_recsys_pipeline.utils.pylogger import RankedLogger

log = RankedLogger(__name__)


class YouTubeDNN(torch.nn.Module):
    def __init__(
        self,
        user_features: list[BaseFeature],
        item_features: list[BaseFeature],
        mlp: Callable,
        temperature: float = 1.0,
        *args,
        **kwargs,
    ):
        """
        Args:
            mlp: A ``functools.partial`` (or Hydra ``_partial_: true``) of
                :class:`~taobao_recsys_pipeline.basic.layers.MLP` that accepts
                ``input_dim`` as its only remaining positional argument.  The
                ``input_dim`` is computed at construction time as
                ``sum(fea.embed_dim for fea in user_features)``.
        """
        log.info(f"Passing in args: {args} and kwargs: {kwargs}.")
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.temperature = temperature
        self.user_dim = sum([fea.embed_dim for fea in user_features])
        self.embedding = EmbeddingLayer(user_features + item_features)
        self.user_mlp = mlp(input_dim=self.user_dim)  # inject computed dim

        self.mode = None

    def set_mode(self, mode: Literal["user", "item"]):
        if mode not in ["user", "item"]:
            raise ValueError(f"Mode must be either 'user' or 'item', but got {mode}")
        self.mode = mode

    def user_tower(self, x: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """
        Returns:
            [B, user_dim] if mode is 'user' else [B, 1, user_dim]
        """
        if self.mode == "item":
            log.warning("Model is in 'item' mode, but user_tower is called. This may lead to incorrect behavior.")
            return None
        user_input = self.embedding(x, self.user_features, squeeze_dim=True)  # [B, user_dim]
        t = self.user_mlp(user_input).unsqueeze(1)  # [B, 1, user_dim]
        t = F.normalize(t, p=2, dim=-1)  # L2 normalize
        if self.mode == "user":
            return t.squeeze(1)  # [B, user_dim] for inference
        return t  # [B, 1, user_dim] for training

    def item_tower(self, x: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """
        Returns:
            [B, user_dim] if mode is 'item' else [B, 1, user_dim]
        """
        if self.mode == "user":
            log.warning("Model is in 'user' mode, but item_tower is called. This may lead to incorrect behavior.")
            return None
        item_input = self.embedding(x, self.item_features, squeeze_dim=True)  # [B, item_dim]
        t = F.normalize(item_input, p=2, dim=-1).unsqueeze(1)  # L2 normalize; [B, 1, item_dim]
        if self.mode == "item":
            return t.squeeze(1)  # [B, item_dim] for inference
        return t  # [B, 1, item_dim] for training

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor | None:
        user_embedding = self.user_tower(x)
        item_embedding = self.item_tower(x)
        if self.mode == "user":
            return user_embedding  # [B, user_dim]
        elif self.mode == "item":
            return item_embedding  # [B, item_dim]

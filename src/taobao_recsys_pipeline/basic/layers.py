from __future__ import annotations

import torch
import torch.nn as nn

from taobao_recsys_pipeline.basic.activation import get_activation_layer
from taobao_recsys_pipeline.basic.features import BaseFeature, SequenceFeature, SparseFeature
from taobao_recsys_pipeline.utils.pylogger import RankedLogger

log = RankedLogger(__name__)


class InputMask(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: dict[str, torch.Tensor], feature: SequenceFeature) -> torch.Tensor:
        if not isinstance(feature, SequenceFeature):
            raise TypeError(f"InputMask only supports SequenceFeature, got {type(feature).__name__}")

        if feature.padding_idx is not None:
            return (x[feature.name].long() != feature.padding_idx).unsqueeze(1).float()  # [B, 1, L]
        else:
            Warning(f"Feature {feature.name} does not have padding_idx, using -1 as default padding_idx")
            return (x[feature.name].long() != -1).unsqueeze(1).float()  # [B, 1, L]


class ConcatPooling(nn.Module):
    """Keep original sequence embedding shape.

    Shape
    -----
    Input: ``(B, L, D)``
    Output: ``(B, L, D)``
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        return x


class AveragePooling(nn.Module):
    """Mean pooling over sequence embeddings.

    Shape
    -----
    Input
        x : ``(B, L, D)``
        mask : ``(B, 1, L)``
    Output
        ``(B, 1, D)``
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return torch.mean(x, dim=1, keepdim=True)  # [B, 1, D]
        else:
            sum_pooling_matrix = torch.bmm(mask, x).squeeze(1)  # [B, D]
            non_padding_length = mask.sum(dim=-1)  # [B, 1]
            return (sum_pooling_matrix / (non_padding_length.float() + 1e-16)).unsqueeze(1)  # [B, 1, D]


class SumPooling(nn.Module):
    """Sum pooling over sequence embeddings.

    Shape
    -----
    Input
        x : ``(B, L, D)``
        mask : ``(B, 1, L)``
    Output
        ``(B, 1, D)``
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return torch.sum(x, dim=1, keepdim=True)  # [B, 1, D]
        else:
            return torch.bmm(mask, x).squeeze(1).unsqueeze(1)  # [B, 1, D]


class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        features: list[BaseFeature],
    ) -> None:
        super().__init__()
        self.features = features
        self.embed_dict = nn.ModuleDict()
        self.input_mask = InputMask()

        for fea in features:
            if fea.shared_with is not None:
                # Embedding table will be borrowed from another feature at forward time;
                # do not create a duplicate entry here.
                continue
            if fea.name in self.embed_dict:
                continue
            self.embed_dict[fea.name] = fea.get_embedding_layer()

    def _get_pooling_layer(self, pooling: str) -> nn.Module:
        pooling_map: dict[str, type[nn.Module]] = {
            "mean": AveragePooling,
            "sum": SumPooling,
            "concat": ConcatPooling,
        }
        if pooling not in pooling_map:
            raise ValueError(f"Pooling method {pooling} not supported")
        return pooling_map[pooling]()

    def _embed_one(
        self,
        fea: SparseFeature | SequenceFeature,
        x: dict[str, torch.Tensor],
        embed_key: str,
    ) -> torch.Tensor:
        """Embed a single feature; returns ``[B, 1 or L, D]``."""
        log.info(
            f"Embedding Feature '{fea.name}' with key '{embed_key}'\n**max value: {x[fea.name].max()}  **min value: {x[fea.name].min()}"
        )
        if isinstance(fea, SparseFeature):
            return self.embed_dict[embed_key](x[fea.name].long()).unsqueeze(1)  # [B, 1, D]
        # SequenceFeature
        emb = self.embed_dict[embed_key](x[fea.name].long())  # [B, L, D]
        fea_mask = self.input_mask(x, fea)  # [B, 1, L]
        return self._get_pooling_layer(fea.pooling)(emb, mask=fea_mask)  # [B, 1 or L, D]

    def forward(
        self,
        x: dict[str, torch.Tensor],
        features: list[BaseFeature] | None = None,
        squeeze_dim: bool = False,
    ):
        """Embed a subset (or all) features and optionally flatten.

        Parameters
        ----------
        x:
            Batch dict mapping feature names to tensors.
        features:
            Subset of features to embed. Defaults to ``self.features`` (all).
        squeeze_dim:
            If ``True``, flatten ``[B, N, D]`` → ``[B, N*D]``.

        Returns
        -------
        torch.Tensor
            ``[B, N*D]`` when *squeeze_dim* is True, else ``[B, N, D]``.
        """
        active_features = features if features is not None else self.features
        sparse_emb = []
        for fea in active_features:
            if fea.name not in x:
                raise ValueError(f"Input x must contain the key '{fea.name}'")
            embed_key = fea.shared_with if fea.shared_with is not None else fea.name
            if embed_key not in self.embed_dict:
                raise KeyError(
                    f"Feature '{fea.name}' references shared embedding '{embed_key}', "
                    "but that key is not registered in EmbeddingLayer. "
                    "Ensure the source feature is included in the full feature list."
                )
            if not isinstance(fea, (SparseFeature, SequenceFeature)):
                raise TypeError(f"Unsupported feature type: {type(fea).__name__}")
            sparse_emb.append(self._embed_one(fea, x, embed_key))

        if not sparse_emb:
            raise ValueError("No features were embedded.")

        sparse_emb_cat = torch.cat(sparse_emb, dim=1)  # [B, N, D]
        if squeeze_dim:
            sparse_emb_cat = sparse_emb_cat.flatten(start_dim=1)  # [B, N*D]
        return sparse_emb_cat


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_layer: bool = True,
        dims: list[int] | None = None,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        if dims is None:
            dims = []

        layers: list[nn.Module] = []
        for i_dim in dims:
            layers.append(nn.Linear(input_dim, i_dim))
            layers.append(nn.BatchNorm1d(i_dim))
            layers.append(get_activation_layer(activation))
            layers.append(nn.Dropout(dropout))
            input_dim = i_dim

        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self.mlp(x)

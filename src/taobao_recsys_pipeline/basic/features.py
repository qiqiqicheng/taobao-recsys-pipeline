import json
from typing import Literal

import torch

from taobao_recsys_pipeline.basic.initializers import RandomNormal


class BaseFeature:
    def __init__(
        self,
        name: str,
        vocab_size: int,
        embed_dim: int,
        padding_idx: int = 0,
        initializer=RandomNormal(0, 0.0001),
        shared_with: str | None = None,
    ):
        self.name = name
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.initializer = initializer
        self.shared_with = shared_with  # name of another feature whose embedding table is reused

    def __repr__(self):
        return f"<BaseFeature {self.name} with Embedding shape ({self.vocab_size}, {self.embed_dim})>"

    def get_embedding_layer(self) -> torch.nn.Embedding:
        if not hasattr(self, "embed"):
            self.embed = self.initializer(self.vocab_size, self.embed_dim, padding_idx=self.padding_idx)
        return self.embed


class SequenceFeature(BaseFeature):
    def __init__(
        self,
        name: str,
        vocab_size: int,
        embed_dim: int,
        pooling: Literal["mean", "sum", "concat"] = "mean",
        padding_idx: int = 0,
        initializer=RandomNormal(0, 0.001),
        edges_file: str | None = None,
        shared_with: str | None = None,
    ):
        self.name = name
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pooling = pooling
        self.padding_idx = padding_idx
        self.initializer = initializer
        self.edges_file = edges_file
        self.shared_with = shared_with

    def __repr__(self):
        return f"<SequenceFeature {self.name} with Embedding shape ({self.vocab_size}, {self.embed_dim})>"

    def get_embedding_layer(self) -> torch.nn.Embedding:
        if not hasattr(self, "embed"):
            self.embed = self.initializer(self.vocab_size, self.embed_dim, padding_idx=self.padding_idx)
        return self.embed


class SparseFeature(BaseFeature):
    def __init__(
        self,
        name: str,
        vocab_size: int,
        embed_dim: int,
        padding_idx: int = 0,
        initializer=RandomNormal(0, 0.0001),
        shared_with: str | None = None,
    ):
        self.name = name
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.initializer = initializer
        self.shared_with = shared_with

    def __repr__(self):
        return f"<SparseFeature {self.name} with Embedding shape ({self.vocab_size}, {self.embed_dim})>"

    def get_embedding_layer(self) -> torch.nn.Embedding:
        if not hasattr(self, "embed"):
            self.embed = self.initializer(self.vocab_size, self.embed_dim, padding_idx=self.padding_idx)
        return self.embed


def get_hist_features(
    file: str,
    embed_dim: int | dict[str, int],
    pooling: str = "mean",
) -> list[SequenceFeature]:
    """Load ``SequenceFeature`` list for the user tower (``hist_*`` fields).

    Reads ``sequence_features.json`` produced by ``prepare_data.py`` and wraps
    each entry as a :class:`SequenceFeature`.  Intended as a Hydra ``_target_``
    so that feature lists can be constructed declaratively in YAML configs.

    Parameters
    ----------
    file:
        Path to ``sequence_features.json``.
    embed_dim:
        Shared embedding dimension (int) or per-feature mapping ``{name: dim}``.
    pooling:
        Default pooling strategy for every feature (can be overridden per-feature
        via ``embed_dim`` dict if needed).
    """
    with open(file) as f:
        configs = json.load(f)
    return [
        SequenceFeature(
            **cfg,
            embed_dim=embed_dim if isinstance(embed_dim, int) else embed_dim[cfg["name"]],
            pooling=pooling,
        )
        for cfg in configs
    ]


def get_target_features(
    file: str,
    embed_dim: int | dict[str, int],
) -> list[SparseFeature]:
    """Build item-tower ``SparseFeature`` list from ``item_sparse_features.json``.

    Each entry (e.g. ``item_id``, ``cat_id``) is:
    * renamed to ``target_{name}`` to match batch keys from :class:`TaobaoDataset`
    * assigned ``shared_with="hist_{name}"`` so that :class:`EmbeddingLayer`
      reuses the embedding table registered for the user-tower sequence feature,
      ensuring the two towers operate in the same embedding space.

    Parameters
    ----------
    file:
        Path to ``item_sparse_features.json`` produced by ``prepare_data.py``.
    embed_dim:
        Shared embedding dimension (int) or per-feature mapping keyed by
        *hist* name (e.g. ``{"hist_item_id": 64}``).
    """
    with open(file) as f:
        configs = json.load(f)
    result: list[SparseFeature] = []
    for cfg in configs:
        target_name = f"target_{cfg['name']}"  # item_id  → target_item_id
        hist_name = f"hist_{cfg['name']}"  # item_id  → hist_item_id
        dim = embed_dim if isinstance(embed_dim, int) else embed_dim.get(hist_name, embed_dim)
        result.append(
            SparseFeature(
                name=target_name,
                vocab_size=cfg["vocab_size"],
                embed_dim=dim,
                shared_with=hist_name,
            )
        )
    return result

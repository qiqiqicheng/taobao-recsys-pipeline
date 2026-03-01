from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from taobao_recsys_pipeline.basic.features import BaseFeature
from taobao_recsys_pipeline.utils.pylogger import RankedLogger
from taobao_recsys_pipeline.basic.layers import EmbeddingLayer, MLP

log = RankedLogger(__name__)


class DIN(nn.Module):
    def __init__(
        self,
        features: list[BaseFeature],
        history_features: list[BaseFeature],
        target_features: list[BaseFeature],
        mlp: Callable,

    )

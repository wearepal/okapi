from dataclasses import dataclass

import torch.nn as nn

from src.models.base import ModelFactoryOut, PredictorFactory

__all__ = ["Identity"]


@dataclass
class Identity(PredictorFactory):
    def __call__(
        self,
        in_dim: int,
    ) -> ModelFactoryOut:
        return nn.Identity(), in_dim

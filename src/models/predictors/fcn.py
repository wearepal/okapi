from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from src.models.base import ModelFactoryOut, PredictorFactory

__all__ = ["Fcn"]


class LayerNormNoBias(nn.Module):
    beta: Parameter

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.gamma = Parameter(torch.ones(input_dim))
        self.register_buffer("beta", torch.zeros(input_dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(
            x,
            normalized_shape=x.shape[-1:],
            weight=self.gamma,
            bias=self.beta,
        )


class NormType(Enum):
    BN = nn.BatchNorm1d
    LN = LayerNormNoBias


class Activation(Enum):
    GELU = nn.GELU
    RELU = nn.ReLU
    SWISH = nn.SiLU


@dataclass
class Fcn(PredictorFactory):
    num_hidden: int = 0
    hidden_dim: Optional[int] = None
    norm: NormType = NormType.LN
    activation: Activation = Activation.GELU
    dropout_prob: float = 0.0

    def __call__(self, in_dim: int, *, out_dim: int) -> ModelFactoryOut:
        predictor = nn.Sequential(nn.Flatten())
        act = self.activation.value()
        curr_dim = in_dim
        if self.num_hidden > 0:
            hidden_dim = in_dim if self.hidden_dim is None else self.hidden_dim
            for _ in range(self.num_hidden):
                predictor.append(nn.Linear(curr_dim, hidden_dim))
                predictor.append(self.norm.value(hidden_dim))
                predictor.append(act)
                if self.dropout_prob > 0:
                    predictor.append(nn.Dropout(p=self.dropout_prob))
                curr_dim = hidden_dim

        predictor.append(nn.Linear(curr_dim, out_dim))
        return predictor, out_dim

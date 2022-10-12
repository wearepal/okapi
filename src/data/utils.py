from typing import Any, Dict, List, Tuple, Union

from conduit.data.datasets.utils import infer_sample_cls
from conduit.data.structures import NamedSample
import torch
from torch import Tensor

__all__ = ["sample_converter", "compute_iw"]


def sample_converter(sample: Union[Any, Tuple[Any, ...], List[Any], Dict[str, Any]]) -> NamedSample:
    sample_cls = infer_sample_cls(sample)
    if isinstance(sample, (tuple, list)):
        sample_d = dict(zip(["y", "s"], sample[1:]))
        return sample_cls(x=sample[0], **sample_d)
    return sample_cls(sample)


@torch.no_grad()
def compute_iw(labels: Tensor) -> Tensor:
    _, inverse, counts = labels.flatten().unique(return_counts=True, return_inverse=True)
    counts_r = counts.reciprocal()
    weights = counts_r / counts_r.sum()
    return weights[inverse]

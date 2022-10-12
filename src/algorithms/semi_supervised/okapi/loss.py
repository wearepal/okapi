import math

import torch
from torch import Tensor
import torch.nn.functional as F

__all__ = [
    "jsd_loss",
    "kld_loss",
]


def jsd_loss(logits_p: Tensor, *, logits_q: Tensor) -> Tensor:
    logits = torch.stack((logits_p, logits_q), dim=0)
    log_probs = F.log_softmax(logits, dim=-1)
    log_m = log_probs.logsumexp(dim=0, keepdim=True) - math.log(2)
    return F.kl_div(
        input=log_probs,
        target=log_m.expand_as(log_probs),
        log_target=True,
        reduction="sum",
    ) / logits.size(0)


def kld_loss(logits_p: Tensor, *, logits_q: Tensor) -> Tensor:
    log_probs_p = F.log_softmax(logits_p, dim=-1)
    log_probs_q = F.log_softmax(logits_q, dim=-1)
    return F.kl_div(
        input=logits_p,
        target=log_probs_q,
        log_target=True,
        reduction="batchmean",
    )

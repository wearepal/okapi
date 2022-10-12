from typing import Callable, Literal, Optional, Type, TypeVar, Union, cast

import torch
from torch import Tensor
import torch.nn as nn
from typing_extensions import Self

__all__ = [
    "DecoupledContrastiveLoss",
    "moco_loss",
    "nnclr_loss",
    "nnclr_loss_v1",
    "nnclr_loss_v2",
    "scl_loss",
]


def moco_loss(
    anchors: Tensor,
    *,
    positives: Tensor,
    negatives: Tensor,
    temperature: float = 1.0,
    dcl: bool = True,
) -> Tensor:
    # positive logits: (N,)
    l_pos = (anchors * positives.unsqueeze(1)).sum(-1).view(-1, 1) / temperature
    # negative logits: (N, K)
    l_neg = (anchors @ negatives.T).view(l_pos.size(0), -1) / temperature
    # Compute the partition function either according to the original InfoNCE formulation
    # or according to the DCL formulation which excludes the positive samples.
    z = l_neg.logsumexp(dim=1) if dcl else torch.cat([l_pos, l_neg], dim=1).logsumexp(dim=1)
    return (z - l_pos).mean()


T = TypeVar("T", Tensor, None)


def scl_loss(
    anchors: Tensor,
    *,
    anchor_labels: Tensor,
    candidates: T = None,
    candidate_labels: T = None,
    temperature: float = 0.1,
    exclude_diagonal: bool = False,
) -> Tensor:
    # Create new variables for the candidate- variables to placate
    # the static-type checker.
    if candidates is None:
        candidates_t = anchors
        candidate_labels_t = anchor_labels
        # Forbid interactions between the samples and themsleves.
        exclude_diagonal = True
    else:
        candidates_t = candidates
        candidate_labels_t = cast(Tensor, candidate_labels)

    anchor_labels = anchor_labels.view(-1, 1)
    candidate_labels_t = candidate_labels_t.flatten()
    # The positive samples for a given anchor are those samples from the candidate set sharing its
    # label.
    mask = anchor_labels == candidate_labels_t
    if exclude_diagonal:
        mask.fill_diagonal_(False)
    pos_inds = mask.nonzero(as_tuple=True)
    row_inds, col_inds = pos_inds
    # Return early if there are no positive pairs.
    if len(row_inds) == 0:
        return anchors.new_zeros(())
    # Only compute the pairwise similarity for those samples which have positive pairs.
    selected_rows, row_inverse, row_counts = row_inds.unique(
        return_inverse=True, return_counts=True
    )
    logits = anchors[selected_rows] @ candidates_t.T
    # Apply temperature-scaling to the logits.
    logits /= temperature
    z = logits.logsumexp(dim=-1).flatten()
    # Tile the row counts if dealing with multicropping.
    if anchors.ndim == 3:
        row_counts = row_counts.unsqueeze(1).expand(-1, anchors.size(1))
    counts_flat = row_counts[row_inverse].flatten()
    positives = logits[row_inverse, ..., col_inds].flatten() / counts_flat
    return (z.sum() - positives.sum()) / z.numel()


def _von_mises_fisher_weighting(z1: Tensor, z2: Tensor, *, sigma: float = 0.5) -> Tensor:
    return 2 - len(z1) * ((z1 * z2).sum(dim=1) / sigma).softmax(dim=0).squeeze()


class DecoupledContrastiveLoss(nn.Module):
    """
    Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf
    """

    def __init__(
        self,
        temperature: float = 0.1,
        *,
        weight_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ):
        """
        :param weight: The weighting function of the positive sample loss.
        :param temperature: Temperature to control the sharpness of the distribution.
        """
        super().__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        """
        Calculate one-way DCL loss

        :param z1: first embedding vector
        :param z2: second embedding vector
        :return: one-way decoupled contrastive loss
        """
        cross_view_distance = torch.mm(z1, z2.t())
        positive_loss = -torch.diag(cross_view_distance) / self.temperature
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(z1, z2)
        neg_similarity = (
            torch.cat((torch.mm(z1, z1.t()), cross_view_distance), dim=1) / self.temperature
        )
        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)
        eps = torch.finfo(z1.dtype).eps
        negative_loss = torch.logsumexp(neg_similarity + neg_mask * eps, dim=1, keepdim=False)
        return (positive_loss + negative_loss).mean()

    @classmethod
    def with_vmf_weighting(
        cls: Type[Self], sigma: float = 0.5, *, temperature: float = 0.1
    ) -> Self:
        return cls(temperature=temperature, weight_fn=_von_mises_fisher_weighting)


def nnclr_loss_v1(
    queries: Tensor, *, keys: Tensor, nn_indices: Tensor, temperature: Union[float, Tensor] = 0.1
) -> Tensor:
    """
    NNCLR loss with softmax computed over the queries, thus matching the formualation given in
    `<https://arxiv.org/abs/2104.14548>`_

    :param queries: Positive (neighbouring) samples w.r.t. the neighbours indicated by nn_indices.
        This variant of the nnclr loss, in essence, treats the neighbours as the anchor samples and the
        queries from which they were computed as the corresponding positive samples.
    :param keys: Pool of samples from which the nearest-neighbours were drawn.
        Expected shape: (M, d)
    :nn_indices: Nearest-neighbour indices. Expected shape: (N,)
    :param temperature: Temperature for the softmax distribution. If ``temperature`` is of type
        Tensor it must contain only a single element.

    :returns: The nearest-neighbour contrastive loss.
    """
    nn_indices = nn_indices.view(len(queries))
    unique_inds, counts = nn_indices.unique(return_counts=True)
    pw_sim = keys[unique_inds] @ queries.T / temperature
    # the positive loss for each is the average similarity w.r.t. its neighbours.
    positive_loss = pw_sim.diag()
    negative_loss = pw_sim.logsumexp(dim=1, keepdim=False)
    return (counts * (negative_loss - positive_loss)).sum() / len(queries)


def nnclr_loss_v2(
    queries: Tensor, *, keys: Tensor, nn_indices: Tensor, temperature: Union[float, Tensor] = 0.1
) -> Tensor:
    """
    NearestNeighbor Contrastive Learning of visual Representations (NNCLR) loss
    with softmax computed over the keys, thus treating the queries as the anchor samples
    and their nearest neighbours, indicated by ``knn_indices``, as the correspoding positive samples.

    :param queries: Anchor samples from which the k-nearest neigbours indicated by ``knn_indices`` were
        computed. Expected shape: (N, d)
    :param keys: Pool of samples from which the nearest-neighbours were drawn. Expected shape: (M, d)
    :knn_indices: Indices indicating the k-nearest neighbours retrieeved for each sample in ``anchors``.
        Expected shape: (N, k)
    :param temperature: Temperature for the softmax distribution. If ``temperature`` is of type
        Tensor it must contain only a single element.

    :returns: The nearest-neighbour contrastive loss.
    """
    nn_indices = nn_indices.view(queries.size(0), -1)  # [N, k]
    pw_sim = queries @ keys.T / temperature  # [N, M]
    # the positive loss for each is the average similarity w.r.t. its neighbours.
    positive_loss = (pw_sim.gather(dim=1, index=nn_indices)).mean(dim=1)
    # the negative loss is simply the usual partition function computed over the key dimension
    negative_loss = pw_sim.logsumexp(dim=1, keepdim=False)
    return (negative_loss - positive_loss).mean()


def nnclr_loss(
    queries: Tensor,
    *,
    keys: Tensor,
    nn_indices: Tensor,
    anchor: Literal["query", "key"] = "query",
    temperature: Union[float, Tensor] = 0.1,
) -> Tensor:
    """
    Nearest-neighbours contrastive loss.

    :param queries: Query samples, Expected shape: (N, d)
    :param keys: Pool of samples from which the nearest-neighbours were drawn. Expected shape: (M, d)
    :nn_indices: Nearest-neighbour indices. Expected shape: (N,)
    :param temperature: Temperature for the softmax distribution. If ``temperature`` is of type
        Tensor it must contain only a single element.
    :anchor: Whether to treat the queries ("query") or keys ("keys") as the anchors for the
        contrastive loss.

    :returns: The nearest-neighbour contrastive loss.
    """
    if anchor == "query":
        return nnclr_loss_v2(
            queries=queries, keys=keys, nn_indices=nn_indices, temperature=temperature
        )
    return nnclr_loss_v1(queries=queries, keys=keys, nn_indices=nn_indices, temperature=temperature)

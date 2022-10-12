from __future__ import annotations
from abc import abstractmethod
from functools import reduce
import math
import operator
from typing import List, NamedTuple, Optional, TypeVar, Union, cast, overload

import attr
from conduit.data.structures import BinarySample, NamedSample
import faiss  # type: ignore
import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from typing_extensions import Literal, Self

from src.algorithms.base import Algorithm
from src.data.datamodules.base import WILDSDataModule

__all__ = [
    "Knn",
    "KnnEvaluator",
    "KnnExact",
    "KnnIVF",
    "KnnIVFPQ",
    "pnorm",
]


def pnorm(
    tensor_a: Tensor,
    tensor_b: Tensor,
    *,
    p: float = 2,
    root: bool = True,
    dim: int = -1,
) -> Tensor:
    dists = (tensor_a - tensor_b).abs()
    if math.isinf(p):
        if p > 0:
            norm = dists.max(dim).values
        else:
            norm = dists.min(dim).values
    else:
        norm = (dists**p).sum(dim)
        if root:
            norm = norm ** (1 / p)  # type: ignore
    return norm


T = TypeVar("T", Tensor, npt.NDArray[np.floating])


class KnnOutput(NamedTuple):
    indices: Tensor | npt.NDArray[np.uint]
    distances: Tensor | npt.NDArray[np.floating]


@attr.define(kw_only=True, eq=False)
class Knn(nn.Module):
    k: int
    p: float = 2
    root: bool = False
    normalize: bool = False
    """
    Whether to Lp-normalize the vectors for pairwise-distance computation.
    .. note:: 
        When vectors u and v are normalized to unit length, the Euclidean distance betwen them
        is equal to :math:`\\|u - v\\|^2 = 2(1-\\cos(u, v))`, that is the Euclidean distance over
        the end-points of u and v is a proper metric which gives the same ordering as the cosine 
        distance for any comparison of vectors, and furthermore avoids the potentially expensive
        trigonometric operations required to yield a proper metric.
    """

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    @abstractmethod
    def _build_index(self, d: int) -> faiss.IndexFlat:
        ...

    def _index_to_gpu(self, index: faiss.IndexFlat) -> faiss.GpuIndexFlat:  # type: ignore
        # use a single GPU
        res = faiss.StandardGpuResources()  # type: ignore
        # make it a flat GPU index
        return faiss.index_cpu_to_gpu(res, x.device.index, index)  # type: ignore

    @overload
    def forward(
        self,
        keys: Tensor,
        *,
        queries: Tensor | None = ...,
        return_distances: Literal[False] = ...,
    ) -> Tensor:
        ...

    @overload
    def forward(
        self,
        keys: Tensor,
        *,
        queries: Tensor | None = ...,
        return_distances: Literal[True] = ...,
    ) -> KnnOutput:
        ...

    def forward(
        self,
        keys: Tensor,
        *,
        queries: Tensor | None = None,
        return_distances: bool = False,
    ) -> Tensor | KnnOutput:

        keys_np = keys.detach().cpu().numpy()
        if self.normalize:
            keys = F.normalize(keys, dim=1, p=self.p)

        if queries is None:
            queries = keys
            queries_np = keys_np
        else:
            if self.normalize:
                queries = F.normalize(queries, dim=1, p=self.p)
            queries_np = queries.detach().cpu().numpy()

        index = self._build_index(d=keys.size(1))
        if keys.is_cuda or queries.is_cuda:
            index = self._index_to_gpu(index=index)

        if not index.is_trained:
            index.train(x=keys_np)  # type: ignore
        # add vectors to the index
        index.add(x=queries_np)  # type: ignore
        # search for the nearest k neighbors for each data-point
        distances_np, indices_np = index.search(x=keys_np, k=self.k)  # type: ignore
        # Convert back from numpy to torch
        indices = torch.as_tensor(indices_np, device=keys.device).squeeze()

        if return_distances:
            if keys.requires_grad or queries.requires_grad:
                distances = pnorm(keys[:, None], queries[indices, :], dim=-1, p=self.p, root=False)
            else:
                distances = torch.as_tensor(distances_np, device=keys.device)

            # Take the root of the distances to 'complete' the norm
            if self.root and (not math.isinf(self.p)):
                distances = distances ** (1 / self.p)

            return KnnOutput(indices=indices, distances=distances)
        return indices


@attr.define(kw_only=True, eq=False)
class KnnExact(Knn):
    def _build_index(self, d: int) -> faiss.IndexFlat:
        index = faiss.IndexFlat(d, faiss.METRIC_Lp)
        index.metric_arg = self.p
        return index


@attr.define(kw_only=True, eq=False)
class KnnIVF(KnnExact):
    nlist: int = 100
    """Number of Voronoi cells to form with k-means clustering."""
    nprobe: int = 1
    """Number of neighboring Voronoi cells to probe."""

    def _build_index(self, d: int) -> faiss.IndexIVFFlat:
        quantizer = super()._build_index(d=d)
        index = faiss.IndexIVFFlat(quantizer, d, self.nlist)
        index.nprobe = self.nprobe
        return index


@attr.define(kw_only=True, eq=False)
class KnnIVFPQ(KnnExact):
    nlist: int = 100
    """Number of Voronoi cells to form with k-means clustering."""
    nprobe: int = 1
    """Number of neighboring Voronoi cells to probe."""
    bits: int = 8
    num_centroids = 8

    def _build_index(self, d: int) -> faiss.IndexIVFPQ:
        quantizer = super()._build_index(d=d)
        m = d // self.num_centroids
        index = faiss.IndexIVFPQ(quantizer, d, self.nlist, m, self.bits)
        index.nprobe = self.nprobe
        return index


class KnnEvaluator(nn.Module):
    keys: Optional[BinarySample]

    def __init__(
        self,
        knn: Knn,
        *,
        datamodule: WILDSDataModule,
    ) -> None:
        super().__init__()
        self.knn = knn
        self.datamodule = datamodule

    def fit(self, pl_module: Algorithm, *, trainer: pl.Trainer, key_dataloader: DataLoader) -> Self:
        keys_ls = trainer.predict(model=pl_module, dataloaders=key_dataloader)
        keys_ls = cast(List[BinarySample[Tensor]], keys_ls)
        keys = reduce(operator.add, keys_ls)
        self.keys = keys
        return self

    @overload
    def predict(
        self, pl_module: Algorithm, *, trainer: pl.Trainer, query_dataloader: DataLoader
    ) -> Tensor:
        ...

    @overload
    def predict(
        self, pl_module: Algorithm, *, trainer: pl.Trainer, query_dataloader: List[DataLoader]
    ) -> List[Tensor]:
        ...

    def predict(
        self,
        pl_module: Algorithm,
        *,
        trainer: pl.Trainer,
        query_dataloader: Union[List[DataLoader], DataLoader],
    ) -> Union[Tensor, List[Tensor]]:
        keys = self.keys
        if keys is None:
            cls_name = self.__class__.__name__
            raise AttributeError(
                f"`{cls_name}.predict` can not be called until a database has been set (this can be"
                f" done by calling '{cls_name}.fit')."
            )
        if isinstance(query_dataloader, list):
            return [
                self.predict(pl_module=pl_module, trainer=trainer, query_dataloader=dl)
                for dl in query_dataloader
            ]
        queries_ls = trainer.predict(model=pl_module, dataloaders=query_dataloader)
        queries_ls = cast(List[NamedSample], queries_ls)
        queries = reduce(operator.add, queries_ls)
        indices = self.knn.forward(keys=keys.x, queries=queries, return_distances=False)
        return keys.y[indices]

    def clear(self) -> None:
        self.keys = None

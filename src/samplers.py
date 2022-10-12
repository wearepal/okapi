from __future__ import annotations
from typing import Iterator, Sequence, Union

from ranzen.decorators import implements
from ranzen.torch.data import BatchSamplerBase, _check_generator
from ranzen.torch.sampling import batched_randint
import torch
from torch import Tensor

__all__ = [
    "HierarchicalSampler",
]


class HierarchicalSampler(BatchSamplerBase):
    def __init__(
        self,
        group_ids: Union[Tensor, Sequence[int]],
        *,
        batch_size: int,
        uniform: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.generator = generator
        self.group_ids = torch.as_tensor(group_ids, dtype=torch.long)
        self.groups, inv_idxs, self.counts = self.group_ids.unique(
            return_inverse=True, return_counts=True
        )
        self.num_groups = len(self.groups)
        # Store the sort-indexes for mapping from group-relative indexes to absolute indexes.
        self._sort_idxs = inv_idxs.sort(dim=0, descending=False).indices
        self._offsets = torch.cat((self.counts.new_zeros(1), self.counts.cumsum(dim=0)[:-1]))

        self.uniform = uniform
        if self.uniform:
            # Sample each group with equal probability.
            self._weights = None
        else:
            # Sample each group with probability proportional to its frequency.
            self._weights = self.counts / self.counts.sum()

        super().__init__(epoch_length=None)

    @implements(BatchSamplerBase)
    def __iter__(self) -> Iterator[list[int]]:
        generator = _check_generator(self.generator)
        # Iterate until some stopping criterion is reached
        while True:
            if self._weights is None:
                group_ids = torch.randint(
                    low=0,
                    high=self.num_groups,
                    size=(self.batch_size,),
                    generator=generator,
                )
            else:
                group_ids = torch.multinomial(
                    self._weights,
                    num_samples=self.batch_size,
                    replacement=True,
                    generator=generator,
                )
            rel_idxs = batched_randint(high=self.counts[group_ids], generator=self.generator)
            abs_idxs = self._sort_idxs[rel_idxs + self._offsets[group_ids]]
            agi = self.group_ids[abs_idxs]
            yield abs_idxs.tolist()

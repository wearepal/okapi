from typing import Dict, List, Optional, Tuple, Union, cast, overload
import warnings

import numpy as np
from ranzen.misc import gcopy
import torch
from torch import Tensor
import torch.nn as nn
from typing_extensions import Literal
from wilds.common.grouper import Grouper  # type: ignore
from wilds.common.utils import get_counts  # type: ignore
from wilds.datasets.wilds_dataset import WILDSDataset  # type: ignore

__all__ = ["CombinatorialGrouper"]


class CombinatorialGrouper(Grouper, nn.Module):
    def __init__(
        self,
        dataset: Union[WILDSDataset, List[WILDSDataset]],
        *,
        groupby_fields: Optional[Union[List[str], Tuple[str], str]],
    ) -> None:
        """
        :param dataset: WILDSDataset(s) from which to extract the metadata.
        :param groupby_fields: Metadata fields to group by.

        CombinatorialGroupers form groups by taking all possible combinations of the metadata
        fields specified in groupby_fields, in lexicographical order.
        For example, if:
            dataset.metadata_fields = ['country', 'time', 'y']
            groupby_fields = ['country', 'time']
        and if in dataset.metadata, country is in {0, 1} and time is in {0, 1, 2},
        then the grouper will assign groups in the following way:
            country = 0, time = 0 -> group 0
            country = 1, time = 0 -> group 1
            country = 0, time = 1 -> group 2
            country = 1, time = 1 -> group 3
            country = 0, time = 2 -> group 4
            country = 1, time = 2 -> group 5

        If groupby_fields is None, then all data points are assigned to group 0.
        """
        nn.Module.__init__(self)
        if isinstance(dataset, (list, tuple)):
            if len(dataset) == 0:
                raise ValueError(
                    f"At least one dataset must be defined for {self.__class__.__name__}."
                )
            datasets = dataset
        else:
            datasets = [dataset]

        datasets = cast(List[WILDSDataset], datasets)
        metadata_fields = datasets[0].metadata_fields
        # Build the largest metadata_map to see to check if all the metadata_maps are subsets of each other
        largest_metadata_map = cast(
            Dict[str, Union[List, np.ndarray]], gcopy(datasets[0].metadata_map, deep=True)
        )
        for i, dataset in enumerate(datasets):
            # The first dataset was used to get the metadata_fields and initial metadata_map
            if i == 0:
                continue

            if dataset.metadata_fields != metadata_fields:
                raise ValueError(
                    f"The datasets passed in have different metadata_fields: {dataset.metadata_fields}. "
                    f"Expected: {metadata_fields}"
                )

            if dataset.metadata_map is None:
                continue
            for field, values in dataset.metadata_map.items():
                n_overlap = min(len(values), len(largest_metadata_map[field]))
                if not (
                    np.asarray(values[:n_overlap])
                    == np.asarray(largest_metadata_map[field][:n_overlap])
                ).all():
                    raise ValueError(
                        "The metadata_maps of the datasets need to be ordered subsets of each other."
                    )

                if len(values) > len(largest_metadata_map[field]):
                    largest_metadata_map[field] = values

        self.groupby_fields = groupby_fields
        if self.groupby_fields is None:
            self._n_groups = 1
        else:
            self.groupby_field_indices = [
                i for (i, field) in enumerate(metadata_fields) if field in groupby_fields
            ]
            if len(self.groupby_field_indices) != len(self.groupby_fields):
                raise ValueError("At least one group field not found in dataset.metadata_fields")

            metadata_array = torch.cat([dataset.metadata_array for dataset in datasets])
            grouped_metadata = metadata_array[:, self.groupby_field_indices]
            if not isinstance(grouped_metadata, torch.LongTensor):
                grouped_metadata_long = grouped_metadata.long()
                if not torch.all(grouped_metadata == grouped_metadata_long):
                    warnings.warn(
                        f'CombinatorialGrouper: converting metadata with fields [{", ".join(self.groupby_fields)}] into long'
                    )
                grouped_metadata = grouped_metadata_long

            for idx, field in enumerate(self.groupby_fields):
                min_value = grouped_metadata[:, idx].min()
                if min_value < 0:
                    raise ValueError(
                        f"Metadata for CombinatorialGrouper cannot have values less than 0: {field}, {min_value}"
                    )
                if min_value > 0:
                    warnings.warn(
                        f"Minimum metadata value for CombinatorialGrouper is not 0 ({field}, {min_value}). This will result in empty groups"
                    )

            # We assume that the metadata fields are integers,
            # so we can measure the cardinality of each field by taking its max + 1.
            # Note that this might result in some empty groups.
            assert grouped_metadata.min() >= 0, "Group numbers cannot be negative."
            self.cardinality = 1 + torch.max(grouped_metadata, dim=0).values
            cumprod = torch.cumprod(self.cardinality, dim=0)
            self._n_groups = int(cumprod[-1].item())
            self.factors_np = np.concatenate([np.array([1]), cumprod[:-1]])
            self.register_buffer("factors", torch.as_tensor(self.factors_np, dtype=torch.double))
            self.metadata_map = largest_metadata_map

    @overload
    def metadata_to_group(
        self, metadata: Tensor, return_counts: Literal[True]
    ) -> Tuple[Tensor, Tensor]:
        ...

    @overload
    def metadata_to_group(self, metadata: Tensor, return_counts: Literal[False] = ...) -> Tensor:
        ...

    def metadata_to_group(
        self, metadata: Tensor, return_counts: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if self.groupby_fields is None:
            groups = torch.zeros(metadata.shape[0], dtype=torch.long)
        else:
            groups = (
                (metadata.view(len(metadata), -1)[:, self.groupby_field_indices]).double()
                @ self.factors
            ).long()

        if return_counts:
            group_counts = get_counts(groups, self.n_groups)
            return groups, group_counts
        return groups

    def group_str(self, group: str) -> str:
        if self.groupby_fields is None:
            return "all"

        # group is just an integer, not a Tensor
        n = len(self.factors_np)
        metadata = np.zeros(n)
        for i in range(n - 1):
            metadata[i] = (group % self.factors_np[i + 1]) // self.factors_np[i]
        metadata[n - 1] = group // self.factors_np[n - 1]
        group_name = ""
        for i in reversed(range(n)):
            meta_val = int(metadata[i])
            if self.metadata_map is not None:
                if self.groupby_fields[i] in self.metadata_map:
                    meta_val = self.metadata_map[self.groupby_fields[i]][meta_val]
            group_name += f"{self.groupby_fields[i]} = {meta_val}, "
        group_name = group_name[:-2]
        return group_name

    def group_field_str(self, group: str) -> str:
        return self.group_str(group).replace("=", ":").replace(",", "_").replace(" ", "")

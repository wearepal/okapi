from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from conduit.data.datasets.utils import PillowTform
from conduit.data.structures import InputContainer, NamedSample, SizedDataset
from ranzen.decorators import implements
from ranzen.misc import gcopy
import torch
from torch import Tensor
from typing_extensions import Self, TypeAlias
from wilds.datasets.unlabeled.wilds_unlabeled_dataset import (  # type: ignore
    WILDSUnlabeledSubset,
)
from wilds.datasets.wilds_dataset import WILDSSubset  # type: ignore

from src.data.utils import sample_converter

if TYPE_CHECKING:
    from src.algorithms.semi_supervised.okapi.matching import (
        MatchedIndices,
        MatchedIndicesBD,
    )

__all__ = [
    "DatasetWrapper",
    "IndexedDataset",
    "IndexedSample",
    "MatchedDataset",
    "MatchedDatasetBD",
    "MatchedSample",
]

Dataset: TypeAlias = Union[WILDSSubset, WILDSUnlabeledSubset, "DatasetWrapper"]
D = TypeVar("D", bound=Union[WILDSSubset, WILDSUnlabeledSubset, "DatasetWrapper"])
D2 = TypeVar("D2", bound=Union[WILDSSubset, WILDSUnlabeledSubset, "DatasetWrapper"])
R_co = TypeVar("R_co", bound=InputContainer, covariant=True)

TextTform: TypeAlias = Callable[[str], Any]
Tform: TypeAlias = Union[PillowTform, TextTform]


class DatasetWrapper(SizedDataset, Generic[R_co, D]):
    dataset: D

    @implements(SizedDataset)
    def __getitem__(self, index: int) -> R_co:
        ...

    @implements(SizedDataset)
    def __len__(self) -> int:
        ...

    @property
    def transform(self) -> Optional[Tform]:
        return self.dataset.transform

    @transform.setter
    def transform(self, value: Optional[Tform]) -> None:
        self.dataset.transform = value


S = TypeVar("S", bound=NamedSample)


@dataclass
class IndexedSample(InputContainer[S]):
    sample: S
    index: Union[Tensor, int]

    @implements(InputContainer)
    def __len__(self) -> int:
        return len(self.sample) + 1

    @implements(InputContainer)
    def __add__(self, other: Self) -> Self:
        copy = gcopy(self, deep=False)
        copy.sample += other.sample
        copy.index = torch.cat((torch.as_tensor(copy.index), torch.as_tensor(other.index)), dim=0)
        return copy


class IndexedDataset(DatasetWrapper[IndexedSample, Dataset]):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __getitem__(self, index: int) -> IndexedSample:
        sample = sample_converter(self.dataset[index])
        return IndexedSample(sample=sample, index=index)

    def __len__(self) -> int:
        return len(self.dataset)


A = TypeVar("A", bound=NamedSample[Tensor])
M = TypeVar("M", bound=NamedSample[Tensor])


@dataclass
class MatchedSample(InputContainer, Generic[A, M]):
    anchor: A
    match: M

    @property
    def x(self) -> Tensor:
        return torch.cat((self.anchor.x, self.match.x), dim=0)

    def split_x(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return tuple(x.split([len(self.anchor.x), len(self.match.x)], dim=0))

    def __post_init__(self) -> None:
        if len(self.anchor.x) != len(self.match.x):
            raise AttributeError("'s1' and 's2' must match in size at dimension 0.")

    @implements(InputContainer)
    def __len__(self) -> int:
        return len(self.anchor.x)

    @implements(InputContainer)
    def __add__(self, other: Self) -> Self:
        copy = gcopy(self, deep=False)
        copy.anchor += other.anchor
        copy.match += other.match
        return copy


@dataclass
class MatchedDatasetBD(Generic[D, D2]):
    tc: "MatchedDataset[D, D2]"
    ct: "MatchedDataset[D2, D]"


class MatchedDataset(DatasetWrapper, Generic[D, D2]):
    def __init__(
        self,
        anchor_dataset: D,
        *,
        matched_dataset: D2,
        anchor_indices: List[int],
        match_indices: List[int],
    ) -> None:
        if len(anchor_indices) != len(match_indices):
            raise AttributeError(
                "'anchor_indices' and 'match_indices' must have equal length (every index in`"
                "'anchor_indices' should be paired with an index in 'match_indices') "
            )
        self.dataset = anchor_dataset
        self.anchor_indices = anchor_indices
        self.match_source = matched_dataset
        self.match_indices = match_indices

    @implements(DatasetWrapper)
    def __getitem__(self, index: int) -> MatchedSample:
        index_a = self.anchor_indices[index]
        anchor = sample_converter(self.dataset[index_a])
        index_m = self.match_indices[index]
        match = sample_converter(self.match_source[index_m])
        return MatchedSample(anchor=anchor, match=match)

    @implements(InputContainer)
    def __len__(self) -> int:
        return len(self.match_indices)

    @classmethod
    def from_matched_indices(
        cls: Type[Self], anchor_dataset: D, *, matched_dataset: D2, indices: "MatchedIndices"
    ) -> "MatchedDataset[D, D2]":
        return MatchedDataset(
            anchor_dataset=anchor_dataset,
            matched_dataset=matched_dataset,
            # ensure indices are flattened
            anchor_indices=indices.anchor.reshape(-1).tolist(),
            match_indices=indices.match.reshape(-1).tolist(),
        )

    @classmethod
    def from_matched_indices_bd(
        cls: Type[Self],
        treatment_dataset: D,
        *,
        control_dataset: D2,
        indices: "MatchedIndicesBD",
    ) -> MatchedDatasetBD[D, D2]:
        md_tc = cls.from_matched_indices(
            anchor_dataset=treatment_dataset, matched_dataset=control_dataset, indices=indices.tc
        )
        md_ct = cls.from_matched_indices(
            anchor_dataset=control_dataset, matched_dataset=treatment_dataset, indices=indices.ct
        )
        return MatchedDatasetBD(tc=md_tc, ct=md_ct)

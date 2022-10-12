from dataclasses import dataclass
from typing import (
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    overload,
)

from conduit.data.structures import InputContainer, LoadedData, concatenate_inputs
from conduit.types import MetricDict
from ranzen.misc import AddDict, gcopy
import torch
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from typing_extensions import Self, TypeAlias, runtime_checkable

__all__ = [
    "BertInput",
    "DictContainer",
    "DistilBertInput",
    "EvalEpochOutput",
    "EvalOutputs",
    "Evaluator",
    "LabeledUnlabeledDlPair",
    "PartiallyLabeledBatch",
]


@runtime_checkable
class Evaluator(Protocol):
    def __call__(self, y_pred: Tensor, y_true: Tensor, metadata: Tensor) -> Tuple[MetricDict, str]:
        ...


TL = TypeVar("TL", bound=InputContainer)
TU = TypeVar("TU", bound=InputContainer)


@runtime_checkable
class PartiallyLabeledBatch(Generic[TL, TU], Protocol):
    @overload
    def __getitem__(self, key: Literal["labeled"]) -> TL:
        ...

    @overload
    def __getitem__(self, key: Literal["unlabeled"]) -> TU:
        ...

    def __getitem__(self, key: Literal["labeled", "unlabeled"]) -> Union[TL, TU]:
        ...

    @overload
    def __setitem__(
        self,
        key: Literal["labeled"],
        value: TL,
    ) -> None:
        ...

    @overload
    def __setitem__(
        self,
        key: Literal["unlabeled"],
        value: TU,
    ) -> None:
        ...

    def __setitem__(
        self,
        key: Literal["labeled", "unlabeled"],
        value: Union[TL, TU],
    ) -> None:
        ...


LabeledUnlabeledDlPair: TypeAlias = Dict[str, DataLoader]


@dataclass
class EvalOutputs(InputContainer):
    logits: Tensor
    targets: Tensor
    metadata: Tensor

    def __len__(self) -> int:
        return len(self.logits)

    def __add__(self, other: Union[Self, int]) -> Self:
        if isinstance(other, int):
            return self
        copy = gcopy(self, deep=False)
        copy.logits = torch.cat((copy.logits, other.logits))
        copy.targets = torch.cat((copy.targets, other.targets))
        copy.metadata = torch.cat((copy.metadata, other.metadata))
        return copy


EvalStepOutput: TypeAlias = AddDict[str, EvalOutputs]
EvalEpochOutput: TypeAlias = List[EvalStepOutput]


class DistilBertInput(TypedDict):
    input_ids: Tensor
    attention_mask: Tensor


class BertInput(DistilBertInput):
    token_type_ids: Tensor


_VT = TypeVar("_VT", bound=LoadedData)


class DictContainer(dict[str, _VT], InputContainer):
    def __add__(
        self: Self,
        other: Union[int, Self],
    ) -> Self:
        # Allow ``other`` to be an integer, but specifying the identity function, for compatibility
        # with th 'no-default' version of``sum``.
        if isinstance(other, int):
            return self
        copy = DictContainer()
        copy.update(gcopy(self, deep=False))
        for key_o, value_o in other.items():
            if key_o in self:
                value_s = self[key_o]
                copy[key_o] = concatenate_inputs(x1=value_s, x2=value_o, is_batched=True)
            else:
                copy[key_o] = value_o
        return copy

    def to(
        self: Self,
        device: Optional[Union[torch.device, str]],
        *,
        non_blocking: bool = False,
    ) -> Self:
        for name, value in self.items():
            if isinstance(value, Tensor):
                self[name] = value.to(device, non_blocking=non_blocking)
        return self

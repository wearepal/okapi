from dataclasses import dataclass
from typing import Any, Optional, Union

from conduit.data.structures import BinarySample
from conduit.models.utils import prefix_keys
from conduit.types import Loss, Stage
from ranzen.decorators import implements
from torch import Tensor
from typing_extensions import TypeAlias

from src.algorithms.base import Algorithm
from src.types import DictContainer, PartiallyLabeledBatch
from src.utils import to_item

from .loss import default_supervised_loss

__all__ = ["Erm"]

TrainBatch: TypeAlias = Union[
    PartiallyLabeledBatch[BinarySample[Tensor], Any],
    BinarySample[Any],
    PartiallyLabeledBatch[BinarySample[DictContainer[Tensor]], Any],
    BinarySample[DictContainer[Tensor]],
]


@dataclass(unsafe_hash=True)
class Erm(Algorithm):
    loss_fn: Optional[Loss] = None

    def _compute_loss(self, logits: Tensor, *, batch: BinarySample[Any]) -> Tensor:
        if self.loss_fn is None:
            return default_supervised_loss(input=logits, target=batch.y)
        return self.loss_fn(input=logits, target=batch.y)

    @implements(Algorithm)
    def training_step(self, batch: TrainBatch, batch_idx: int) -> Tensor:
        batch = batch if isinstance(batch, BinarySample) else batch["labeled"]
        logits = self.forward(batch.x)
        loss = self._compute_loss(logits=logits, batch=batch)

        results_dict = {"batch_loss": to_item(loss)}
        results_dict = prefix_keys(dict_=results_dict, prefix=str(Stage.fit), sep="/")
        self.log_dict(results_dict)

        return loss

from dataclasses import dataclass
from typing import Optional, Tuple

from conduit.data.structures import BinarySample, NamedSample
from conduit.models.utils import prefix_keys
from conduit.types import Loss, MetricDict, Stage
from ranzen.decorators import implements
from ranzen.torch.loss import ReductionType, cross_entropy_loss
import torch
from torch import Tensor
from typing_extensions import TypeAlias

from src.algorithms.semi_supervised.base import SemiSupervisedAlgorithm
from src.transforms import FixMatchPair
from src.types import PartiallyLabeledBatch
from src.utils import to_item

__call__ = ["FixMatch", "fixmatch_consistency_loss"]

TrainBatch: TypeAlias = PartiallyLabeledBatch[
    BinarySample[Tensor], NamedSample[FixMatchPair[Tensor]]
]


def fixmatch_consistency_loss(
    logits_weak: Tensor,
    *,
    logits_strong: Tensor,
    temperature: float,
    confidence_threshold: float,
) -> Tuple[Tensor, float]:
    # Generate the supra-threshold psuedo-labels from the logits of the weakly-augmented
    # version of the image.
    with torch.no_grad():
        tempered_probs = (logits_weak / temperature).softmax(dim=1).detach()
        max_res = tempered_probs.max(dim=1)
        max_probs, pseudo_labels = max_res.values, max_res.indices
        pseudo_label_mask = max_probs >= confidence_threshold

    if num_retained := pseudo_label_mask.count_nonzero().item():
        # Compute the loss using the above-generated psuedo-labels as targets for the
        # strongly-augmented images.
        loss = cross_entropy_loss(
            input=logits_strong[pseudo_label_mask],
            target=pseudo_labels[pseudo_label_mask],
            reduction=ReductionType.sum,
        ) / len(logits_weak)
    else:
        loss = logits_weak.new_zeros(())
    retention_rate = num_retained / len(pseudo_label_mask)

    return loss, retention_rate


@dataclass(unsafe_hash=True)
class FixMatch(SemiSupervisedAlgorithm):

    confidence_threshold: float = 0.95
    loss_fn: Optional[Loss] = None
    loss_u_weight: float = 1.0
    temperature: float = 1.0
    soft: bool = False

    def __post_init__(self) -> None:
        if self.confidence_threshold < 0:
            raise AttributeError("'confidence_threshold' must be in the range [0, 1].")
        if self.loss_u_weight < 0:
            raise AttributeError("'loss_u_weight' must be non-negative.")
        if self.temperature < 0:
            raise AttributeError("'temperature' must be non-negative.")

    @implements(SemiSupervisedAlgorithm)
    def training_step(self, batch: TrainBatch, batch_idx: int) -> Tensor:
        batch_l = batch["labeled"]
        x_l = batch["labeled"].x
        logging_dict: MetricDict = {}
        retention_rate = None
        if self.loss_u_weight > 0:
            batch_u = batch["unlabeled"]
            x_u_strong = batch_u.x.strong
            x_u_weak = batch_u.x.weak
            x_all = torch.cat([x_l, x_u_weak, x_u_strong], dim=0)
            logits = self.model.forward(x_all)
            logits_l, logits_u_weak, logits_u_strong = logits.split(
                [len(x_l), len(x_u_weak), len(x_u_strong)], dim=0
            )
            if self.soft:
                loss_u = cross_entropy_loss(
                    input=logits_u_strong, target=logits_u_weak.detach().softmax(dim=0)
                )
            else:
                # Pseudo-labels can currently only be computed for classification tasks.
                if batch_l.y.is_floating_point():
                    loss_u, retention_rate = fixmatch_consistency_loss(
                        logits_weak=logits_u_weak,
                        logits_strong=logits_u_strong,
                        temperature=self.temperature,
                        confidence_threshold=self.confidence_threshold,
                    )
                else:
                    # For regression tasks the consistency loss is simply the MSE
                    # between the predictions for the weakly- and strongly-augmented inputs.
                    loss_u = self._compute_supervised_loss(
                        input=logits_u_weak, target=logits_u_strong
                    )
            loss_u *= self.loss_u_weight
            logging_dict["consistency"] = to_item(loss_u)
        else:
            logits_l = self.model.forward(x_l)
            loss_u = 0.0
        # Compute the supervised loss using the labeled data.
        loss_s = self._compute_supervised_loss(input=logits_l, target=batch_l.y)
        loss = loss_s + loss_u

        logging_dict["supervised"] = to_item(loss_s)
        logging_dict["total"] = to_item(loss)
        logging_dict = prefix_keys(
            dict_=logging_dict,
            prefix=f"{str(Stage.fit)}/batch_loss",
            sep="/",
        )
        if retention_rate is not None:
            logging_dict[f"{str(Stage.fit)}/pseudo_label_retention_rate"] = retention_rate

        self.log_dict(logging_dict)

        return loss

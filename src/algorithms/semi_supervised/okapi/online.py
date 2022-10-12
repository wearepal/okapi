from dataclasses import dataclass, field
from enum import Enum, auto
from typing import ClassVar, Optional, TypeVar, Union

from conduit.data.structures import BinarySample, SubgroupSample, TernarySample
from conduit.metrics import accuracy
from conduit.models.utils import prefix_keys
from conduit.types import MetricDict, Stage
from ranzen.decorators import implements
from ranzen.torch.loss import ReductionType, cross_entropy_loss
from ranzen.torch.schedulers import LinearWarmup, WarmupScheduler
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import TypeAlias

from src.algorithms.loss import default_supervised_loss
from src.algorithms.mean_teacher import MeanTeacher
from src.algorithms.memory_bank import MemoryBank
from src.algorithms.self_supervised.loss import nnclr_loss
from src.algorithms.semi_supervised.base import SemiSupervisedAlgorithm
from src.data.utils import compute_iw
from src.models import FeaturesLogits
from src.types import DictContainer, PartiallyLabeledBatch
from src.utils import to_item

from .matching import BinaryCaliperNN, CaliperNN, LrSolver

__all__ = ["Okapi"]

X = TypeVar("X", bound=Union[Tensor, DictContainer])

TrainBatch: TypeAlias = PartiallyLabeledBatch[
    TernarySample[X],
    BinarySample[X],
]


class ConsistencyLoss(Enum):
    KL = auto()
    L2_FEAT = auto()
    NNCLR = auto()


@dataclass(unsafe_hash=True)
class Okapi(SemiSupervisedAlgorithm):
    IGNORE_INDEX: ClassVar[int] = -100

    fixed_caliper_max: float = 0.95
    std_caliper: Optional[float] = 0.2
    twoway_caliper: bool = True
    temp_ps: float = 1.0
    reweight: bool = True
    c: float = 1.0
    solver: LrSolver = LrSolver.SAGA
    normalize: bool = False
    k: int = 1

    ema_decay_start: float = 0.9
    ema_decay_end: float = 0.999
    ema_warmup_steps: int = 0
    loss_u_fn: ConsistencyLoss = ConsistencyLoss.KL
    temp_nnclr: float = 0.1

    mb_capacity: int = 16_384
    warmup_steps: int = 0
    online_ps: bool = True
    multilabel_ps: bool = False
    binary: bool = False
    use_oe_for_queries: bool = False

    loss_u_weight_start: float = 0.0
    loss_u_weight_end: float = 1.0
    warmup_steps: int = 0
    loss_u_weight: WarmupScheduler = field(init=False)

    matcher: Union[CaliperNN, BinaryCaliperNN] = field(init=False)
    ema_model: Optional[MeanTeacher] = field(init=False)
    feature_mb: MemoryBank = field(init=False)
    logit_mb: Optional[MemoryBank] = field(init=False)
    propensity_scorer: Optional[nn.Linear] = field(init=False)

    def __post_init__(self) -> None:
        if self.temp_ps <= 0:
            raise AttributeError("'temperature' must be positive.")
        if self.loss_u_weight_start < 0:
            raise AttributeError("'loss_u_weight_start' must be non-negative.")
        if self.loss_u_weight_end < 0:
            raise AttributeError("'loss_u_weight_end' must be non-negative.")
        if not (0 <= self.ema_decay_start < 1):
            raise AttributeError("'ema_decay_start' must be in the range [0, 1).")
        if not (0 <= self.ema_decay_end <= 1):
            raise AttributeError("'ema_decay_end' must be in the range [0, 1].")
        if self.warmup_steps < 0:
            raise AttributeError("'warmup_steps' must be non-negative.")

        if self.loss_u_fn is ConsistencyLoss.NNCLR:
            self.normalize = True

        matcher_cls = BinaryCaliperNN if self.binary else CaliperNN
        self.matcher = matcher_cls(
            fixed_caliper_max=self.fixed_caliper_max,
            std_caliper=self.std_caliper,
            twoway_caliper=self.twoway_caliper,
            temperature=self.temp_ps,
            reweight=self.reweight,
            normalize=False,  # normalize externally instead
            solver=self.solver,
            k=self.k,
        )
        if self.ema_decay_end == 0:
            self.ema_model = None
        else:
            self.ema_model = MeanTeacher(
                self.model,
                decay_start=self.ema_decay_start,
                decay_end=self.ema_decay_end,
                warmup_steps=self.ema_warmup_steps,
                auto_update=False,
            )
        self.feature_mb = MemoryBank.with_l2_hypersphere_init(
            dim=self.model.feature_dim, capacity=self.mb_capacity
        )
        self.label_mb = MemoryBank.with_constant_init(
            dim=1, capacity=self.mb_capacity, value=self.IGNORE_INDEX, dtype=torch.long
        )

        if self.loss_u_fn is ConsistencyLoss.KL:
            self.logit_mb = MemoryBank.with_l2_hypersphere_init(
                dim=self.model.out_dim, capacity=self.mb_capacity
            )
        else:
            self.logit_mb = None

        card_s = self.grouper.n_groups
        ps_dim = 1 if self.binary else card_s
        self.propensity_scorer = (
            nn.Linear(
                in_features=self.model.feature_dim,
                out_features=ps_dim,
            )
            if self.online_ps
            else None
        )
        self.loss_u_weight = LinearWarmup(
            start_val=self.loss_u_weight_start,
            end_val=self.loss_u_weight_end,
            warmup_steps=self.warmup_steps,
        )

    @implements(SemiSupervisedAlgorithm)
    def training_step(self, batch: TrainBatch, batch_idx: int) -> Tensor:
        batch_l = batch["labeled"]

        logging_dict: MetricDict = {}
        match_rate = None
        acc_ps = None
        loss = batch_l.y.new_zeros((), dtype=torch.float)

        if self.loss_u_weight.val > 0:
            batch_u = batch["unlabeled"]
            if isinstance(batch_l.x, Tensor):
                x_lu = torch.cat((batch_l.x, batch_u.x), dim=0)
            else:
                x_lu = batch_l.x + batch_u.x
            outputs_q = self.model.forward(x_lu, return_features=True)
            features_on, logits_on = outputs_q.features, outputs_q.logits
            # l2-normalize the encodings, thereby making the matching and consistency loss
            # cosine-distance-based.
            if self.normalize:
                features_on = F.normalize(features_on, dim=1)
            logits_on_l = logits_on[: len(batch_l.y)]

            with torch.no_grad():
                if self.ema_model is None:
                    outputs_tgt = outputs_q
                else:
                    self.ema_model.update()
                    outputs_tgt: FeaturesLogits = self.ema_model.forward(x_lu, return_features=True)

                features_tgt, logits_tgt = outputs_tgt.features, outputs_tgt.logits
                # l2-normalize the encodings, thereby making the matching and consistency loss
                # cosine-distance-based.
                if self.normalize:
                    features_tgt = F.normalize(features_tgt, dim=1)

                queries = features_on.detach() if self.use_oe_for_queries else features_tgt
                if self.binary:
                    labels_l_q = queries.new_ones(len(batch_l.y), dtype=torch.long)
                    labels_u_q = queries.new_zeros(len(batch_u.y), dtype=torch.long)
                else:
                    labels_l_q = self.grouper.metadata_to_group(batch_l.s).long()
                    # The unlabeled data is unlabeled w.r.t. the target and thus consists of duples
                    # where the 'y' label corresponds to the domain indicator.
                    labels_u_q = self.grouper.metadata_to_group(batch_u.y).long()

                labels_q = torch.cat((labels_l_q, labels_u_q))
                mb_mask = (self.label_mb.memory != self.IGNORE_INDEX).squeeze(-1)
                labels_k_mb = self.label_mb[mb_mask].clone().view(-1)
                # Update the domain-label memory bank.
                self.label_mb.push(labels_q)
                # Perform the same union operation as above to form the labels for the keys.
                labels_k = torch.cat((labels_q, labels_k_mb), dim=0)

                # Keys are formed from the union of the queries and the memory-bank-stored features.
                keys = torch.cat((queries, self.feature_mb[mb_mask].clone()), dim=0)
                # Update the feature memory bank.
                self.feature_mb.push(features_tgt)

            if self.propensity_scorer is None:
                logits_ps = None
            else:
                logits_ps = self.propensity_scorer(keys)
                instance_weights = compute_iw(labels_k) if self.reweight else None

                if (not self.binary) and self.multilabel_ps:
                    if instance_weights is None:
                        targets_ps = F.one_hot(labels_k, num_classes=logits_ps.size(1)).float()
                    else:
                        targets_ps = instance_weights.new_zeros(logits_ps.size()).scatter_(
                            1, labels_k.view(-1, 1), instance_weights.view(-1, 1)
                        )
                    loss_ps = F.binary_cross_entropy_with_logits(input=logits_ps, target=targets_ps)
                else:
                    loss_ps = cross_entropy_loss(
                        input=logits_ps,
                        target=labels_k,
                        instance_weight=instance_weights,
                    )
                logging_dict["propensity"] = to_item(loss_ps)
                acc_ps = accuracy(y_pred=logits_ps, y_true=labels_k)
                loss += loss_ps

            with torch.no_grad():
                if logits_ps is None:
                    ps_query, ps_key = None, None
                else:
                    logits_ps /= self.temp_ps
                    if self.multilabel_ps or (logits_ps.size(1) == 1):
                        ps_key = logits_ps.sigmoid()
                    else:
                        ps_key = logits_ps.softmax(dim=1)
                    ps_query = ps_key[: len(features_on)]

                query_sample = SubgroupSample(x=queries, s=labels_q)
                key_sample = SubgroupSample(x=keys, s=labels_k)
                if isinstance(self.matcher, BinaryCaliperNN):
                    matched_indices_bd = self.matcher(
                        queries=query_sample,
                        keys=key_sample,
                        direction=None,  # bidirectional matching
                        ps_query=ps_query,
                        ps_key=ps_key,
                    )
                    matched_indices = matched_indices_bd.tc + matched_indices_bd.ct
                else:
                    matched_indices = self.matcher(
                        queries=query_sample,
                        keys=key_sample,
                        ps_query=ps_query,
                        ps_key=ps_key,
                    )

            if (match_rate := (len(matched_indices) / len(queries))) > 0:
                if self.logit_mb is not None:
                    logits_anchor = logits_on[matched_indices.anchor]
                    logits_match = torch.cat([logits_tgt, self.logit_mb[mb_mask].clone()], dim=0)
                    target_probs = (logits_match[matched_indices.match]).softmax(dim=1)
                    consistency_loss = cross_entropy_loss(
                        input=logits_anchor,
                        target=target_probs,
                        reduction=ReductionType.mean,
                    )
                else:
                    features_anchor = features_on[matched_indices.anchor]
                    if self.loss_u_fn is ConsistencyLoss.L2_FEAT:
                        features_match = keys[matched_indices.match]
                        consistency_loss = (
                            (features_anchor.unsqueeze(1) - features_match).pow(2).sum(-1).mean()
                        )
                    else:
                        consistency_loss = nnclr_loss(
                            queries=features_anchor,
                            keys=keys,
                            nn_indices=matched_indices.match,
                            temperature=self.temp_nnclr,
                            anchor="query",
                        )

                consistency_loss *= match_rate
                consistency_loss *= self.loss_u_weight
                logging_dict["consistency"] = to_item(consistency_loss)
                loss += consistency_loss

            if self.logit_mb is not None:
                self.logit_mb.push(logits_tgt)
        else:
            logits_on_l = self.model.forward(batch_l.x)

        self.loss_u_weight.step()

        loss_s = default_supervised_loss(input=logits_on_l, target=batch_l.y)
        loss += loss_s
        logging_dict["supervised"] = to_item(loss_s)
        logging_dict["total"] = to_item(loss)
        logging_dict = prefix_keys(
            dict_=logging_dict,
            prefix=f"{str(Stage.fit)}/batch_loss",
            sep="/",
        )
        if match_rate is not None:
            logging_dict["match_rate"] = match_rate
        if acc_ps is not None:
            logging_dict["acc_propensity_scorer"] = acc_ps
        self.log_dict(logging_dict)

        return loss

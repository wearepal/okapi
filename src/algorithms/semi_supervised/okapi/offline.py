from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, cast

import attr
from conduit.data.structures import BinarySample, SubgroupSample, TernarySample
from conduit.logging import init_logger
from conduit.models.utils import prefix_keys
from conduit.types import MetricDict, Stage
import pytorch_lightning as pl
from ranzen.decorators import implements
from ranzen.misc import gcopy
from ranzen.torch.schedulers import LinearWarmup, WarmupScheduler
import torch
from torch import Tensor
import torch.nn.functional as F
from typing_extensions import Self, TypeAlias
import wandb

from src.algorithms.base import Algorithm
from src.algorithms.encoder import DatasetEncoder
from src.algorithms.semi_supervised.base import SemiSupervisedAlgorithm
from src.algorithms.semi_supervised.fixmatch import fixmatch_consistency_loss
from src.data.datamodules.base import LabeledDataset, UnlabeledDataset, WILDSDataModule
from src.data.wrappers import MatchedDataset, MatchedDatasetBD, MatchedSample
from src.models.artifact import load_model_from_artifact
from src.types import LabeledUnlabeledDlPair, PartiallyLabeledBatch
from src.utils import to_item

from .artifact import load_indices_from_artifact, save_indices_artifact
from .loss import jsd_loss, kld_loss
from .matching import BinaryCaliperNN, LrSolver

__all__ = ["OkapiOffline"]

TrainBatch: TypeAlias = PartiallyLabeledBatch[
    MatchedSample[TernarySample[Tensor], BinarySample[Tensor]],
    MatchedSample[BinarySample[Tensor], TernarySample[Tensor]],
]

LOGGER = init_logger(name=__file__)


class ConsistencyLoss(Enum):
    JSD = auto()
    KLD = auto()
    FM = auto()
    L2_FEAT = auto()
    COSINE_FEAT = auto()


@dataclass(unsafe_hash=True)
class OkapiOffline(SemiSupervisedAlgorithm):
    # Matching configuration.
    fixed_caliper_min: float = 0.1
    fixed_caliper_max: float = 0.9
    std_caliper: Optional[float] = 0.2
    twoway_caliper: bool = True
    temp_ps: float = 1.0
    reweight: bool = True
    c: float = 1.0
    cosine_dist: bool = False
    solver: LrSolver = LrSolver.SAGA

    # Loss configuration.
    loss_u_fn: ConsistencyLoss = ConsistencyLoss.JSD
    loss_u_weight_start: float = 0.0
    loss_u_weight_end: float = 1.0
    warmup_steps: int = 0
    loss_u_weight: WarmupScheduler = field(init=False)
    confidence_threshold: float = 0.95
    fm_temperature: float = 1.0

    indices_artifact: Optional[str] = None
    encoder_artifact: Optional[str] = None
    artifact_project: Optional[str] = None

    def __post_init__(self) -> None:
        if self.temp_ps <= 0:
            raise AttributeError("'temp_ps' must be positive.")
        if self.fm_temperature <= 0:
            raise AttributeError("'fm_temperature' must be positive.")
        if self.loss_u_weight_start > self.loss_u_weight_end:
            raise AttributeError(
                "'loss_u_weight_start' must be less than or equal to 'loss_u_weight_end'."
            )
        if self.loss_u_weight_start < 0:
            raise AttributeError("'loss_u_weight_start' must be non-negative.")
        if self.loss_u_weight_end < 0:
            raise AttributeError("'loss_u_weight_end' must be non-negative.")
        if self.warmup_steps < 0:
            raise AttributeError("'warmup_steps' must be non-negative.")
        if self.confidence_threshold < 0:
            raise AttributeError("'confidence_threshold' must be in the range [0, 1].")

        self.loss_u_weight = LinearWarmup(
            start_val=self.loss_u_weight_start,
            end_val=self.loss_u_weight_end,
            warmup_steps=self.warmup_steps,
        )

    @implements(SemiSupervisedAlgorithm)
    def training_step(
        self,
        batch: TrainBatch,
        batch_idx: int,
    ) -> Tensor:
        # Supervised loss.
        batch_tc = batch["labeled"]
        batch_ct = batch["unlabeled"]

        x_tc = batch_tc.x
        x_ct = batch_ct.x
        x_all = torch.cat((x_tc, x_ct), dim=0)

        model_out = self.model.forward(x_all, return_features=True)

        logits_tc, logits_ct = model_out.logits.split([len(x_tc), len(x_ct)], dim=0)
        logits_tc_a, logits_tc_m = batch_tc.split_x(logits_tc)
        logits_ct_a, logits_ct_m = batch_ct.split_x(logits_ct)

        logits_l = torch.cat((logits_tc_a, logits_ct_m), dim=0)
        y_all = torch.cat((batch_tc.anchor.y, batch_ct.match.y), dim=0)

        loss = loss_s = self._compute_supervised_loss(input=logits_l, target=y_all)
        logging_dict: MetricDict = {"supervised": to_item(loss_s)}

        retention_rate = None
        if self.loss_u_weight.val > 0:
            if self.loss_u_fn in (ConsistencyLoss.L2_FEAT, ConsistencyLoss.COSINE_FEAT):
                features_tc, features_ct = model_out.features.split([len(x_tc), len(x_ct)], dim=0)
                features_tc_a, features_tc_m = batch_tc.split_x(features_tc)
                features_ct_a, features_ct_m = batch_ct.split_x(features_ct)
                features_a = torch.cat((features_tc_a, features_ct_a), dim=0)
                features_m = torch.cat((features_tc_m, features_ct_m), dim=0)
                # l2-normalize the encodings, thereby making the matching and consistency loss
                # cosine-distance-based.
                if self.loss_u_fn is ConsistencyLoss.COSINE_FEAT:
                    features_a = F.normalize(features_a, dim=1)
                    features_m = F.normalize(features_m, dim=1)
                features_m = features_m.view(
                    len(features_a), -1, features_a.size(1)
                )  # reshape for backwards compatibility
                consistency_loss = (features_a.unsqueeze(1) - features_m).pow(2).sum(-1).mean()
            else:
                logits_a = torch.cat((logits_tc_a, logits_ct_a), dim=0)
                logits_m = torch.cat((logits_tc_m, logits_ct_m), dim=0)
                if self.loss_u_fn is ConsistencyLoss.JSD:
                    consistency_loss = jsd_loss(logits_p=logits_a, logits_q=logits_m)
                elif self.loss_u_fn is ConsistencyLoss.KLD:
                    consistency_loss = kld_loss(logits_p=logits_a, logits_q=logits_m)
                else:
                    consistency_loss, retention_rate = fixmatch_consistency_loss(
                        logits_weak=logits_a,
                        logits_strong=logits_m,
                        temperature=self.temp_ps,
                        confidence_threshold=self.confidence_threshold,
                    )
            consistency_loss *= self.loss_u_weight
            logging_dict["consistency"] = to_item(consistency_loss)
            loss += consistency_loss
        self.loss_u_weight.step()

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

    def generate_matched_datasets(
        self, datamodule: WILDSDataModule, *, trainer: pl.Trainer
    ) -> MatchedDatasetBD[LabeledDataset, UnlabeledDataset]:
        dls = datamodule.train_dataloader(eval=True)
        dls = cast(LabeledUnlabeledDlPair, dls)
        run = wandb.run
        if (run is not None) and (self.indices_artifact is not None):
            matched_indices = load_indices_from_artifact(
                name=self.indices_artifact, run=run, project=self.artifact_project
            )
        else:
            if (run is None) or (self.encoder_artifact is None):
                encoder = self.model.backbone
            else:
                encoder, _ = load_model_from_artifact(
                    name=self.encoder_artifact,
                    run=run,
                    project=self.artifact_project,
                )
            ds_encoder = DatasetEncoder(model=encoder)
            LOGGER.info("Encoding labeled data...")
            features_l_ls = cast(
                List[Tensor],
                trainer.predict(ds_encoder, dataloaders=dls["labeled"], return_predictions=True),
            )
            features_l: Tensor = torch.cat(features_l_ls)
            LOGGER.info("Encoding unlabeled data...")
            features_u_ls = cast(
                List[Tensor],
                trainer.predict(ds_encoder, dataloaders=dls["unlabeled"], return_predictions=True),
            )
            features_u: Tensor = torch.cat(features_u_ls)

            features = torch.cat((features_l, features_u), dim=0)
            labels_l = features_l.new_ones((len(features_l),), dtype=torch.long)
            labels_u = labels_l.new_zeros((len(features_u),), dtype=torch.long)
            labels = torch.cat([labels_l, labels_u], dim=0)
            LOGGER.info("Computing matches...")
            matcher = BinaryCaliperNN(
                fixed_caliper_max=self.fixed_caliper_max,
                std_caliper=self.std_caliper,
                twoway_caliper=self.twoway_caliper,
                temperature=self.temp_ps,
                reweight=self.reweight,
                normalize=self.cosine_dist,
                solver=self.solver,
                k=1,
            )
            matcher_inputs = SubgroupSample(x=features, s=labels)
            matched_indices = matcher(matcher_inputs, direction=None)
            # Remove the offset in the indices due to matching being performed over
            # the concatenation of the labeled and unlabeled data (in that order).
            matched_indices.tc.match -= len(labels_l)
            matched_indices.ct.anchor -= len(labels_l)
            if run is not None:
                if self.encoder_artifact is None:
                    encoder_name = encoder.__class__.__name__.lower()
                    artifact_name = (
                        f"{datamodule.__class__.__name__.removesuffix('DataModule').lower()}_"
                    )
                    artifact_name += encoder_name
                else:
                    artifact_name = encoder_name = self.encoder_artifact.replace(":", "_")
                artifact_name += "_match_indices"

                save_indices_artifact(
                    indices=matched_indices,
                    run=run,
                    name=artifact_name,
                    metadata={"encoder": encoder_name} | attr.asdict(matcher),
                )

        train_data_l = datamodule.train_data.labeled
        train_data_u = cast(UnlabeledDataset, datamodule.train_data.unlabeled)
        return MatchedDataset.from_matched_indices_bd(
            treatment_dataset=train_data_l, control_dataset=train_data_u, indices=matched_indices
        )

    @implements(Algorithm)
    def _run_internal(
        self, datamodule: WILDSDataModule, *, trainer: pl.Trainer, test: bool = True
    ) -> Self:
        matched_datasets = self.generate_matched_datasets(datamodule=datamodule, trainer=trainer)
        # datamodule = gcopy(datamodule, deep=True)
        datamodule.train_data.labeled = matched_datasets.tc
        datamodule.train_data.unlabeled = matched_datasets.ct

        return super()._run_internal(datamodule=datamodule, trainer=trainer, test=test)

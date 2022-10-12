from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import attr
from fairscale.nn import auto_wrap  # type: ignore
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from ranzen.decorators import implements
from ranzen.hydra import Option, Relay

from src.algorithms.base import Algorithm
from src.conf import WandbLoggerConf
from src.data.datamodules.base import WILDSDataModule
from src.models import MetaModel, Model
from src.models.artifact import save_model_artifact

__all__ = ["WILDSRelay"]


@attr.define(kw_only=True)
class WILDSRelay(Relay):
    dm: DictConfig
    alg: DictConfig
    backbone: DictConfig
    predictor: DictConfig
    trainer: DictConfig
    logger: DictConfig
    checkpointer: DictConfig

    meta_model: Optional[DictConfig] = None
    pt_alg: Optional[DictConfig] = None
    pt_trainer: Optional[DictConfig] = None

    seed: Optional[int] = 0
    artifact_name: Optional[str] = None

    @classmethod
    @implements(Relay)
    def with_hydra(
        cls,
        root: Path | str,
        *,
        dm: list[Option],
        alg: list[Option],
        backbone: list[Option],
        predictor: list[Option],
        meta_model: list[Option],
        clear_cache: bool = False,
    ) -> None:

        configs = dict(
            dm=dm,
            alg=alg,
            pt_alg=alg,
            backbone=backbone,
            predictor=predictor,
            meta_model=meta_model,
            trainer=[Option(class_=pl.Trainer, name="base")],
            pt_trainer=[Option(class_=pl.Trainer, name="base")],
            logger=[Option(class_=WandbLoggerConf, name="base")],
            checkpointer=[Option(class_=ModelCheckpoint, name="base")],
        )
        super().with_hydra(
            root=root,
            instantiate_recursively=False,
            clear_cache=clear_cache,
            **configs,
        )

    @implements(Relay)
    def run(self, raw_config: Dict[str, Any]) -> None:
        self.log(f"Current working directory: '{os.getcwd()}'")
        pl.seed_everything(self.seed, workers=True)

        dm: WILDSDataModule = instantiate(self.dm)
        dm.prepare_data()
        dm.setup()

        backbone, feature_dim = instantiate(self.backbone)()
        predictor, out_dim = instantiate(self.predictor)(in_dim=feature_dim, out_dim=dm.target_dim)
        model: Union[Model, MetaModel]
        model = Model(
            backbone=backbone, feature_dim=feature_dim, predictor=predictor, out_dim=out_dim
        )

        # enable parameter sharding with fairscale.
        # Note: when fully-sharded training is not enabled this is a no-op
        model = auto_wrap(model)  # type: ignore

        if self.logger.get("group", None) is None:
            default_group = f"{dm.__class__.__name__.removesuffix('DataModule').lower()}_"
            default_group += "_".join(
                dict_conf["_target_"].split(".")[-1].lower()
                for dict_conf in (self.backbone, self.predictor, self.alg)
            )
            self.logger["group"] = default_group
        logger: WandbLogger = instantiate(self.logger, reinit=True)
        if raw_config is not None:
            logger.log_hyperparams(raw_config)  # type: ignore

        # Disable checkpointing when instantiating the trainer as we want to use
        # a hydra-instantiated checkpointer.
        trainer: pl.Trainer = instantiate(
            self.trainer,
            logger=logger,
            enable_checkpointing=False,
        )
        checkpointer: ModelCheckpoint = instantiate(self.checkpointer)
        trainer.callbacks.append(checkpointer)

        if self.pt_alg is not None:
            pt_logger: WandbLogger = instantiate(
                self.logger, reinit=True, prefix="pretraining", experiment=logger.experiment
            )
            pt_trainer_conf = self.trainer if self.pt_trainer is None else self.pt_trainer
            pt_trainer: pl.Trainer = instantiate(
                pt_trainer_conf,
                logger=pt_logger,
                enable_checkpointing=False,
            )
            pt_trainer.callbacks.append(checkpointer)

            pt_alg: Algorithm = instantiate(
                self.pt_alg,
                model=model,
                evaluator=dm.evaluator,
                grouper=dm.train_grouper,
            )
            pt_alg.run(datamodule=dm, trainer=pt_trainer, test=False)

        if self.meta_model is not None:
            model = instantiate(self.meta_model, _partial_=True)(model=model)
        alg: Algorithm = instantiate(self.alg, _partial_=True)(
            model=model,
            evaluator=dm.evaluator,
            grouper=dm.train_grouper,
        )
        alg.run(datamodule=dm, trainer=trainer)

        if (self.artifact_name is not None) and (not logger.experiment.offline):
            save_model_artifact(
                model=model,
                artifact_name=self.artifact_name,
                run=logger.experiment,
                config=raw_config,
                filename="final_model.pt",
            )

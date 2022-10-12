from dataclasses import dataclass
from typing import Any, Optional, cast

from conduit.logging import init_logger
from conduit.types import Loss
import pytorch_lightning as pl
from ranzen import gcopy
from ranzen.decorators import implements
from torch import Tensor
from torch.utils.data import ConcatDataset
from typing_extensions import Self

from src.algorithms.base import Algorithm
from src.algorithms.loss import default_supervised_loss
from src.data.datamodules import PovertyMapDataModule
from src.data.datamodules.base import UnlabeledDataset, WILDSDataModule
from src.types import PartiallyLabeledBatch

__all__ = ["SemiSupervisedAlgorithm"]

LOGGER = init_logger(name=__file__)


@dataclass(unsafe_hash=True)
class SemiSupervisedAlgorithm(Algorithm):
    loss_fn: Optional[Loss] = None
    use_test_data: bool = False

    def _compute_supervised_loss(self, input: Tensor, *, target: Tensor) -> Tensor:
        if self.loss_fn is None:
            return default_supervised_loss(input=input, target=target)
        return self.loss_fn(input=input, target=target)

    def training_step(self, batch: PartiallyLabeledBatch[Any, Any], batch_idx: int) -> Tensor:
        ...

    @implements(Algorithm)
    def run(self, datamodule: WILDSDataModule, *, trainer: pl.Trainer, test: bool = True) -> Self:
        if not datamodule.use_unlabeled:
            LOGGER.info(
                f"'use_unlabeled' must be 'True' when running with a semi-supervised algorithm."
                " Now resetting the data-module again with 'use_unlabeled=True' forced."
            )
            datamodule.use_unlabeled = True
            datamodule.setup(force_reset=True)

        # # Since the unlabelled training data for poverty map is not disjoint from
        # # the labelled data in terms of domains, the ood validation (and optionally test)
        # # data should be used when, for instance, running CaliperNN in binary mode.
        if isinstance(datamodule, PovertyMapDataModule):
            transform_u = cast(UnlabeledDataset, datamodule.train_data.unlabeled).transform
            train_data_u = cast(
                UnlabeledDataset, gcopy(datamodule.val_data.ood.unlabeled, deep=False)
            )
            train_data_u.transform = transform_u
            if self.use_test_data:
                train_data_u_2 = cast(UnlabeledDataset, gcopy(datamodule.test_data.ood.unlabeled))
                train_data_u = ConcatDataset([train_data_u, train_data_u_2])  # type: ignore
            datamodule.train_data.unlabeled = train_data_u  # type: ignore

        return self._run_internal(datamodule=datamodule, trainer=trainer, test=test)

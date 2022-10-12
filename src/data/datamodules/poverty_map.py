"""Data-module for the Poverty Map dataset."""
from enum import Enum, auto
from typing import Any

import attr
from pytorch_lightning import LightningDataModule
from ranzen import implements
from ranzen.decorators import enum_name_str
import torchvision.transforms as T  # type: ignore
from wilds.datasets.poverty_dataset import PovertyMapDataset  # type: ignore
from wilds.datasets.unlabeled.poverty_unlabeled_dataset import (  # type: ignore
    PovertyMapUnlabeledDataset,
)

from src.data.datamodules.base import (
    EvalSplit,
    Split,
    TrainValTestSplit,
    WILDSVisionDataModule,
)
from src.transforms import Identity, RandAugmentPM

__all__ = [
    "PovertyMapDataModule",
    "PovertyMapFold",
    "PovertyMapSplit",
]


@enum_name_str
class PovertyMapSplit(Enum):
    official = auto()
    "Official split: equivalent to 'time_after_2018'"
    mixed_to_test = auto()


@enum_name_str
class PovertyMapFold(Enum):
    A = auto()
    B = auto()
    C = auto()
    D = auto()
    E = auto()


@attr.define(kw_only=True)
class PovertyMapDataModule(WILDSVisionDataModule):
    """Data-module for the FMoW dataset."""

    use_unlabeled_y: bool = False
    use_ood_val: bool = True
    fold: PovertyMapFold = PovertyMapFold.A
    no_nl: bool = False
    split_scheme: PovertyMapSplit = PovertyMapSplit.official

    @property  # type: ignore[misc]
    @implements(WILDSVisionDataModule)
    def _default_train_transforms(self) -> RandAugmentPM:
        return RandAugmentPM()

    @property  # type: ignore[misc]
    @implements(WILDSVisionDataModule)
    def _default_test_transforms(self) -> Identity:
        return Identity()

    @implements(LightningDataModule)
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        PovertyMapDataset(
            root_dir=self.root,
            split_scheme=str(self.split_scheme),
            no_nl=self.no_nl,
            fold=self.fold.name,
            download=True,
            use_ood_val=self.use_ood_val,
        )
        if self.use_unlabeled:
            PovertyMapUnlabeledDataset(
                root_dir=self.root,
                split_scheme=str(self.split_scheme),
                no_nl=self.no_nl,
                fold=self.fold.name,
                download=True,
                use_ood_val=self.use_ood_val,
            )

    @implements(WILDSVisionDataModule)
    def _get_splits(self) -> TrainValTestSplit:
        labeled_data = PovertyMapDataset(
            root_dir=self.root,
            split_scheme=str(self.split_scheme),
            no_nl=self.no_nl,
            fold=self.fold.name,
            use_ood_val=self.use_ood_val,
        )
        train_data = labeled_data.get_subset(split="train")
        val_data = labeled_data.get_subset(split="val")
        test_data = labeled_data.get_subset(split="test")
        id_val_data = labeled_data.get_subset(split="id_val")  # ID validation set
        id_test_data = labeled_data.get_subset(split="id_test")  # ID test set

        train_data_u = None
        val_data_u = None
        test_data_u = None

        if self.use_unlabeled:
            unlabeled_data = PovertyMapUnlabeledDataset(
                root_dir=self.root,
                split_scheme=str(self.split_scheme),
                no_nl=self.no_nl,
                fold=self.fold.name,
                use_ood_val=self.use_ood_val,
            )

            train_data_u = unlabeled_data.get_subset(
                split="train_unlabeled", load_y=self.use_unlabeled_y
            )
            val_data_u = unlabeled_data.get_subset(
                split="val_unlabeled", load_y=self.use_unlabeled_y
            )
            test_data_u = unlabeled_data.get_subset(
                split="test_unlabeled", load_y=self.use_unlabeled_y
            )

        train_data = Split(labeled=train_data, unlabeled=train_data_u)
        ood_val_data = Split(labeled=val_data, unlabeled=val_data_u)
        ood_test_data = Split(labeled=test_data, unlabeled=test_data_u)

        val_data = EvalSplit(ood=ood_val_data, id=id_val_data)
        test_data = EvalSplit(ood=ood_test_data, id=id_test_data)

        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)

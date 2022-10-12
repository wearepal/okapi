"""Data-module for the FMoW dataset."""
from enum import Enum, auto
from typing import Any

import attr
from conduit.data.constants import IMAGENET_STATS
from conduit.data.datasets.utils import PillowTform
from pytorch_lightning import LightningDataModule
from ranzen import implements
from ranzen.decorators import enum_name_str
import torchvision.transforms as T  # type: ignore
from torchvision.transforms.functional import InterpolationMode  # type: ignore
from wilds.datasets.fmow_dataset import FMoWDataset  # type: ignore
from wilds.datasets.unlabeled.fmow_unlabeled_dataset import (  # type: ignore
    FMoWUnlabeledDataset,
)

from src.data.datamodules.base import (
    EvalSplit,
    Split,
    TrainValTestSplit,
    WILDSVisionDataModule,
)

__all__ = [
    "FMoWDataModule",
    "FMoWSplit",
]


@enum_name_str
class FMoWSplit(Enum):
    official = auto()
    "Official split: equivalent to 'time_after_2018'"
    mixed_to_test = auto()
    time_after_2002 = auto()
    time_after_2003 = auto()
    time_after_2004 = auto()
    time_after_2005 = auto()
    time_after_2006 = auto()
    time_after_2007 = auto()
    time_after_2008 = auto()
    time_after_2009 = auto()
    time_after_2010 = auto()
    time_after_2011 = auto()
    time_after_2012 = auto()
    time_after_2013 = auto()
    time_after_2014 = auto()
    time_after_2015 = auto()
    time_after_2016 = auto()
    time_after_2017 = auto()
    time_after_2018 = auto()


@attr.define(kw_only=True)
class FMoWDataModule(WILDSVisionDataModule):
    """Data-module for the FMoW dataset."""

    use_unlabeled_y: bool = False
    use_ood_val: bool = True
    split_scheme: FMoWSplit = FMoWSplit.official
    target_resolution: int = 224

    @property  # type: ignore[misc]
    @implements(WILDSVisionDataModule)
    def _default_train_transforms(self) -> T.Compose:
        transforms_ls: list[PillowTform] = [
            T.Resize(
                size=(self.target_resolution, self.target_resolution),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.CenterCrop(self.target_resolution),
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=self.target_resolution),
            T.RandAugment(num_ops=2),
            T.ToTensor(),
            T.Normalize(*IMAGENET_STATS),
        ]
        return T.Compose(transforms_ls)

    @property  # type: ignore[misc]
    @implements(WILDSVisionDataModule)
    def _default_test_transforms(self) -> T.Compose:
        transforms_ls: list[PillowTform] = [
            T.Resize(
                size=(self.target_resolution, self.target_resolution),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.CenterCrop(self.target_resolution),
            T.ToTensor(),
            T.Normalize(*IMAGENET_STATS),
        ]
        return T.Compose(transforms_ls)

    @implements(LightningDataModule)
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        FMoWDataset(
            root_dir=self.root,
            split_scheme=str(self.split_scheme),
            download=True,
            use_ood_val=self.use_ood_val,
        )
        if self.use_unlabeled:
            FMoWUnlabeledDataset(
                root_dir=self.root,
                split_scheme=str(self.split_scheme),
                download=True,
                use_ood_val=self.use_ood_val,
            )

    @implements(WILDSVisionDataModule)
    def _get_splits(self) -> TrainValTestSplit:
        labeled_data = FMoWDataset(
            root_dir=self.root, split_scheme=str(self.split_scheme), use_ood_val=self.use_ood_val
        )
        train_data = labeled_data.get_subset(split="train")
        val_data = labeled_data.get_subset(split="val")  # OOD validation set
        test_data = labeled_data.get_subset(split="test")  # OOD test set

        id_val_data = labeled_data.get_subset(split="id_val")  # ID validation set
        id_test_data = labeled_data.get_subset(split="id_test")  # ID test set

        train_data_u = None
        val_data_u = None
        test_data_u = None

        if self.use_unlabeled:
            unlabeled_data = FMoWUnlabeledDataset(
                root_dir=self.root,
                split_scheme=str(self.split_scheme),
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

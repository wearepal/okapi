"""Data-module for the Global Wheat dataset."""
from enum import Enum, auto
from typing import Any

import attr
from conduit.data.constants import IMAGENET_STATS
from conduit.data.datasets.utils import PillowTform
from pytorch_lightning import LightningDataModule
from ranzen import implements
from ranzen.decorators import enum_name_str
import torchvision.transforms as T  # type: ignore
from wilds.datasets.globalwheat_dataset import GlobalWheatDataset  # type: ignore
from wilds.datasets.unlabeled.globalwheat_unlabeled_dataset import (  # type: ignore
    GlobalWheatUnlabeledDataset,
)

from src.data.datamodules.base import (
    EvalSplit,
    Split,
    TrainValTestSplit,
    WILDSVisionDataModule,
)

__all__ = [
    "GlobalWheatDatamodule",
    "GlobalWheatSplit",
]


@enum_name_str
class GlobalWheatSplit(Enum):
    official = auto()
    official_with_subsampled_test = auto()
    fixed_test = auto()
    mixed_train = auto()


@attr.define(kw_only=True)
class GlobalWheatDatamodule(WILDSVisionDataModule):
    """Data-module for the iWildCam dataset."""

    use_unlabeled_y: bool = False
    split_scheme: GlobalWheatSplit = GlobalWheatSplit.official

    @property  # type: ignore[misc]
    @implements(WILDSVisionDataModule)
    def _default_train_transforms(self) -> T.Compose:
        # Although GlobalWheat is an image dataset and can be transformed using augmentations
        # we omit data-augmentations for simplicity, because such augmentations would generally
        # require changing y as well as x (e.g., random translations on the input image also
        # require translating the bounding box labels).
        transforms_ls: list[PillowTform] = [T.ToTensor(), T.Normalize(*IMAGENET_STATS)]
        return T.Compose(transforms_ls)

    @property  # type: ignore[misc]
    @implements(WILDSVisionDataModule)
    def _default_test_transforms(self) -> T.Compose:
        transforms_ls: list[PillowTform] = [T.ToTensor(), T.Normalize(*IMAGENET_STATS)]
        return T.Compose(transforms_ls)

    @implements(LightningDataModule)
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        GlobalWheatDataset(root_dir=self.root, split_scheme=str(self.split_scheme), download=True)
        if self.use_unlabeled:
            GlobalWheatUnlabeledDataset(
                root_dir=self.root, split_scheme=str(self.split_scheme), download=True
            )

    @implements(WILDSVisionDataModule)
    def _get_splits(self) -> TrainValTestSplit:
        labeled_data = GlobalWheatDataset(root_dir=self.root, split_scheme=str(self.split_scheme))
        train_data = labeled_data.get_subset(split="train")
        ood_val_data = labeled_data.get_subset(split="val")
        ood_test_data = labeled_data.get_subset(split="test")

        id_val_data = labeled_data.get_subset(split="id_val")  # ID validation set
        id_test_data = labeled_data.get_subset(split="id_test")  # ID test set

        train_data_u = None
        val_data_u = None
        test_data_u = None

        if self.use_unlabeled:
            unlabeled_data = GlobalWheatUnlabeledDataset(
                root_dir=self.root, split_scheme=str(self.split_scheme)
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
        ood_val_data = Split(labeled=ood_val_data, unlabeled=val_data_u)
        ood_test_data = Split(labeled=ood_test_data, unlabeled=test_data_u)

        test_data = EvalSplit(ood=ood_test_data, id=id_test_data)
        val_data = EvalSplit(ood=ood_val_data, id=id_val_data)

        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)

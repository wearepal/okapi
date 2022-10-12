"""Data-module for the iWildCam dataset."""
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
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset  # type: ignore
from wilds.datasets.unlabeled.camelyon17_unlabeled_dataset import (  # type: ignore
    Camelyon17UnlabeledDataset,
)

from src.data.datamodules.base import (
    EvalSplit,
    Split,
    TrainValTestSplit,
    WILDSVisionDataModule,
)

__all__ = ["Camelyon17DataModule"]


@enum_name_str
class Camelyon17Split(Enum):
    official = auto()
    """Oficial split."""

    mixed_to_test = auto()
    """ 
    For the mixed-to-test setting, slide 23 (corresponding to patient 042, node 3 in the
    original dataset) is moved from the test set to the training set
    """


@attr.define(kw_only=True)
class Camelyon17DataModule(WILDSVisionDataModule):
    """Data-module for the iWildCam dataset."""

    use_unlabeled_y: bool = False
    split_scheme: Camelyon17Split = Camelyon17Split.official
    target_resolution: int = 96

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
        Camelyon17Dataset(root_dir=self.root, split_scheme=str(self.split_scheme), download=True)
        if self.use_unlabeled:
            Camelyon17UnlabeledDataset(
                root_dir=self.root, split_scheme=str(self.split_scheme), download=True
            )

    @implements(WILDSVisionDataModule)
    def _get_splits(self) -> TrainValTestSplit:
        labeled_data = Camelyon17Dataset(root_dir=self.root, split_scheme=str(self.split_scheme))
        train_data = labeled_data.get_subset(split="train")
        val_data = labeled_data.get_subset(split="val")  # OOD validation set
        test_data = labeled_data.get_subset(split="test")  # OOD test set

        id_val_data = labeled_data.get_subset(split="id_val")  # ID validation set

        train_data_u = None
        val_data_u = None
        test_data_u = None

        if self.use_unlabeled:
            unlabeled_data = Camelyon17UnlabeledDataset(
                root_dir=self.root, split_scheme=str(self.split_scheme)
            )
            train_data_u = unlabeled_data.get_subset(
                split="extra_unlabeled", load_y=self.use_unlabeled_y
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
        test_data = EvalSplit(ood=ood_test_data, id=None)

        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)

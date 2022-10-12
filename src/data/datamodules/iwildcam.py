"""Data-module for the iWildCam dataset."""
from typing import Any

import attr
from conduit.data.constants import IMAGENET_STATS
from conduit.data.datasets.utils import PillowTform
from pytorch_lightning import LightningDataModule
from ranzen import implements
import torchvision.transforms as T  # type: ignore
from torchvision.transforms.functional import InterpolationMode  # type: ignore
from wilds.datasets.iwildcam_dataset import IWildCamDataset  # type: ignore
from wilds.datasets.unlabeled.iwildcam_unlabeled_dataset import (  # type: ignore
    IWildCamUnlabeledDataset,
)

from src.data.datamodules.base import (
    EvalSplit,
    Split,
    TrainValTestSplit,
    WILDSVisionDataModule,
)

__all__ = ["IWildCamDataModule"]


@attr.define(kw_only=True)
class IWildCamDataModule(WILDSVisionDataModule):
    """Data-module for the iWildCam dataset."""

    target_resolution: int = 448
    use_unlabeled_y: bool = False

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
        IWildCamDataset(root_dir=self.root, split_scheme="official", download=True)
        if self.use_unlabeled:
            IWildCamUnlabeledDataset(root_dir=self.root, split_scheme="official", download=True)

    @implements(WILDSVisionDataModule)
    def _get_splits(self) -> TrainValTestSplit:
        labeled_data = IWildCamDataset(root_dir=self.root, split_scheme="official")
        train_data_l = labeled_data.get_subset(split="train")
        ood_val_data = Split(labeled=labeled_data.get_subset(split="val"))  # OOD validation set
        ood_test_data = Split(labeled=labeled_data.get_subset(split="test"))  # OOD test set

        id_val_data = labeled_data.get_subset(split="id_val")  # ID validation set
        id_test_data = labeled_data.get_subset(split="id_test")  # ID test set
        val_data = EvalSplit(ood=ood_val_data, id=id_val_data)
        test_data = EvalSplit(ood=ood_test_data, id=id_test_data)

        train_data_u = None
        if self.use_unlabeled:
            unlabeled_data = IWildCamUnlabeledDataset(root_dir=self.root, split_scheme="official")
            train_data_u = unlabeled_data.get_subset(
                split="extra_unlabeled", load_y=self.use_unlabeled_y
            )
        train_data = Split(labeled=train_data_l, unlabeled=train_data_u)

        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)

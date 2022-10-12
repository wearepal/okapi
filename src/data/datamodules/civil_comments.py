"""Data-module for the FMoW dataset."""
from enum import Enum, auto
from typing import Any

import attr
from pytorch_lightning import LightningDataModule
from ranzen import implements
from ranzen.decorators import enum_name_str
from wilds.datasets.civilcomments_dataset import CivilCommentsDataset  # type: ignore
from wilds.datasets.unlabeled.civilcomments_unlabeled_dataset import (  # type: ignore
    CivilCommentsUnlabeledDataset,
)

from src.data.datamodules.base import (
    EvalSplit,
    Split,
    TrainValTestSplit,
    WILDSTextDataModule,
)

__all__ = [
    "CivilCommentsDataModule",
    "CivilCommentsSplit",
]


@enum_name_str
class CivilCommentsSplit(Enum):
    official = auto()


@attr.define(kw_only=True)
class CivilCommentsDataModule(WILDSTextDataModule):
    """Data-module for the FMoW dataset."""

    use_unlabeled_y: bool = False
    use_ood_val: bool = True
    split_scheme: CivilCommentsSplit = CivilCommentsSplit.official

    @implements(LightningDataModule)
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        CivilCommentsDataset(
            root_dir=self.root,
            split_scheme=str(self.split_scheme),
            download=True,
        )
        if self.use_unlabeled:
            CivilCommentsUnlabeledDataset(
                root_dir=self.root,
                split_scheme=str(self.split_scheme),
                download=True,
            )

    @implements(WILDSTextDataModule)
    def _get_splits(self) -> TrainValTestSplit:
        labeled_data = CivilCommentsDataset(root_dir=self.root, split_scheme=str(self.split_scheme))
        train_data = labeled_data.get_subset(split="train")
        val_data = labeled_data.get_subset(split="val")  # OOD validation set
        test_data = labeled_data.get_subset(split="test")  # OOD test set

        ood_val_data = Split(labeled=val_data)
        ood_test_data = Split(labeled=test_data)

        val_data = EvalSplit(ood=ood_val_data)
        test_data = EvalSplit(ood=ood_test_data)

        train_data_u = None
        if self.use_unlabeled:
            unlabeled_data = CivilCommentsUnlabeledDataset(
                root_dir=self.root,
                split_scheme=str(self.split_scheme),
            )

            train_data_u = unlabeled_data.get_subset(
                split="extra_unlabeled", load_y=self.use_unlabeled_y
            )

        train_data = Split(labeled=train_data, unlabeled=train_data_u)

        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)

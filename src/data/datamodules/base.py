from abc import abstractmethod
from enum import Enum, auto
import logging
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import attr
from conduit.data import CdtDataLoader
from conduit.logging import init_logger
from conduit.types import Stage
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
from ranzen.decorators import implements
from ranzen.misc import gcopy
from ranzen.torch import SequentialBatchSampler, StratifiedBatchSampler, TrainingMode
from ranzen.torch.data import TrainingMode, WeightedBatchSampler
from torch import Tensor
from torch.utils.data import Sampler
from torch.utils.data.dataloader import DataLoader
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing_extensions import TypeAlias, final
from wilds.datasets.unlabeled.wilds_unlabeled_dataset import (  # type: ignore
    WILDSUnlabeledSubset,
)
from wilds.datasets.wilds_dataset import WILDSDataset, WILDSSubset  # type: ignore

from src.data.grouper import CombinatorialGrouper
from src.data.utils import sample_converter
from src.data.wrappers import DatasetWrapper
from src.samplers import HierarchicalSampler
from src.transforms import ImageToTensorTransform
from src.types import DictContainer, Evaluator, LabeledUnlabeledDlPair

__all__ = [
    "EvalSplit",
    "Split",
    "TrainValTestSplit",
    "WILDSDataModule",
    "WILDSTextDataModule",
    "WILDSVisionDataModule",
]

LabeledDataset: TypeAlias = Union[WILDSSubset, DatasetWrapper[Any, WILDSSubset]]
UnlabeledDataset = Union[
    WILDSUnlabeledSubset, DatasetWrapper[Any, WILDSUnlabeledSubset], LabeledDataset
]
L = TypeVar("L", bound=LabeledDataset, covariant=True)
U = TypeVar("U", bound=Optional[UnlabeledDataset])


def _get_base_dataset(dataset: Union[LabeledDataset, UnlabeledDataset]) -> WILDSDataset:
    return dataset.dataset if isinstance(dataset, DatasetWrapper) else dataset


@attr.define(kw_only=True)
class Split(Generic[L, U]):
    labeled: L
    unlabeled: U = attr.field(default=None)

    @property
    def y_size(self) -> int:
        return (
            self.labeled.dataset.y_size
            if isinstance(self.labeled, DatasetWrapper)
            else self.labeled.y_size
        )

    @property
    def n_classes(self) -> Optional[int]:
        return (
            self.labeled.dataset.n_classes
            if isinstance(self.labeled, DatasetWrapper)
            else self.labeled.n_classes
        )

    @property
    def num_labeled(self) -> int:
        return len(self.labeled)

    @property
    def num_unlabeled(self) -> int:
        if self.unlabeled is None:
            return 0
        return len(self.unlabeled)

    def __len__(self) -> int:
        return self.num_labeled + self.num_unlabeled

    def __iter__(
        self,
    ) -> Iterator[Union[WILDSSubset, WILDSUnlabeledSubset, DatasetWrapper]]:
        if self.unlabeled is None:
            yield self.labeled
        else:
            yield from (self.labeled, self.unlabeled)


I = TypeVar("I", bound=Optional[WILDSSubset])


@attr.define(kw_only=True)
class EvalSplit(Generic[L, U, I]):
    ood: Split[L, U]
    id: I = attr.field(default=None)

    @property
    def num_ood_samples(self) -> int:
        return len(self.ood)

    @property
    def num_id_samples(self) -> int:
        if self.id is None:
            return 0
        return len(self.id)

    def __len__(self) -> int:
        return self.num_ood_samples + self.num_id_samples

    @property
    def labeled(self) -> Dict[str, LabeledDataset]:
        datasets: Dict[str, LabeledDataset] = {"OOD": self.ood.labeled}
        if self.id is not None:
            datasets["ID"] = self.id
        return datasets

    def __iter__(self) -> Iterator[Union[WILDSSubset, WILDSUnlabeledSubset, DatasetWrapper]]:
        if self.id is None:
            yield from self.ood
        else:
            yield from (*self.ood, self.id)


I2 = TypeVar("I2", bound=Optional[WILDSSubset])


@attr.define(kw_only=True)
class TrainValTestSplit(Generic[L, U, I, I2]):
    train: Split[L, U]
    val: EvalSplit[WILDSSubset, U, I]
    test: EvalSplit[WILDSSubset, U, I2]

    def __iter__(
        self,
    ) -> Iterator[Union[Split[L, U], EvalSplit[WILDSSubset, U, I], EvalSplit[WILDSSubset, U, I2]]]:
        yield from (self.train, self.val, self.test)


class SamplingMethod(Enum):
    stratified = auto()
    standard = auto()
    weighted = auto()
    hierarchical = auto()


@attr.define(kw_only=True)
class WILDSDataModule(pl.LightningDataModule):

    train_batch_size_l: int = 16
    _train_batch_size_u: Optional[int] = None
    _eval_batch_size: Optional[int] = None
    num_workers: int = 0
    seed: int = 47
    persist_workers: bool = False
    pin_memory: bool = True
    sampling_method: SamplingMethod = SamplingMethod.standard
    groupby_fields: Optional[List[str]] = None
    sampling_groups: Optional[List[str]] = None
    training_mode: TrainingMode = TrainingMode.epoch
    use_unlabeled: bool = False

    _logger: Optional[logging.Logger] = attr.field(default=None, init=False)
    _train_data: Optional[Split] = attr.field(default=None, init=False)
    _val_data: Optional[EvalSplit] = attr.field(default=None, init=False)
    _test_data: Optional[EvalSplit] = attr.field(default=None, init=False)
    _evaluator: Optional[Evaluator] = attr.field(default=None, init=False)

    _dim_x: Optional[Tuple[int, ...]] = attr.field(default=None, init=False)
    _card_y: Optional[int] = attr.field(default=None, init=False)
    _dim_y: Optional[int] = attr.field(default=None, init=False)

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = init_logger(self.__class__.__name__)
        return self._logger

    @property
    def train_batch_size_u(self) -> int:
        if self._train_batch_size_u is None:
            return self.train_batch_size_l
        return self._train_batch_size_u

    @property
    def eval_batch_size(self) -> int:
        if self._eval_batch_size is None:
            return self.train_batch_size_l
        return self._eval_batch_size

    @property
    @final
    def train_data(self) -> Split[LabeledDataset, Optional[UnlabeledDataset]]:
        self._check_setup_called()
        return cast(Split, self._train_data)

    @property
    @final
    def val_data(self) -> EvalSplit[WILDSSubset, Optional[UnlabeledDataset], Optional[WILDSSubset]]:
        self._check_setup_called()
        return cast(EvalSplit, self._val_data)

    @property
    @final
    def test_data(
        self,
    ) -> EvalSplit[WILDSSubset, Optional[UnlabeledDataset], Optional[WILDSSubset]]:
        self._check_setup_called()
        return cast(EvalSplit, self._test_data)

    @property
    def train_grouper(self):
        datasets = [_get_base_dataset(ds) for ds in self.train_data]
        return CombinatorialGrouper(dataset=datasets, groupby_fields=self.groupby_fields)

    @property
    def val_grouper(self):
        datasets = [_get_base_dataset(ds) for ds in self.val_data]
        return CombinatorialGrouper(dataset=datasets, groupby_fields=self.groupby_fields)

    @property
    def test_grouper(self):
        datasets = [_get_base_dataset(ds) for ds in self.test_data]
        return CombinatorialGrouper(dataset=datasets, groupby_fields=self.groupby_fields)

    @property
    @final
    def evaluator(self) -> Evaluator:
        self._check_setup_called()
        return cast(Evaluator, self._evaluator)

    @property
    def is_set_up(self) -> bool:
        return self._train_data is not None

    @final
    def _check_setup_called(self, caller: Optional[str] = None) -> None:
        if not self.is_set_up:
            if caller is None:
                # inspect the call stack to find out who called this function
                import inspect

                caller = inspect.getouterframes(inspect.currentframe(), 2)[1][3]

            cls_name = self.__class__.__name__
            raise AttributeError(
                f"'{cls_name}.{caller}' cannot be accessed as '{cls_name}.setup()' has "
                "not yet been called."
            )

    @property
    def dim_x(self) -> Tuple[int, ...]:
        if self._dim_x is None:
            self._check_setup_called()
            input_size = tuple(self._train_data.labeled[0].x.shape)  # type: ignore
            self._dim_x = input_size
        return self._dim_x

    @property
    @final
    def dim_y(self) -> int:
        self._check_setup_called()
        return self.train_data.y_size

    @property
    @final
    def card_y(self) -> int:
        self._check_setup_called()
        card_y = self.train_data.n_classes
        if card_y is None:
            raise AttributeError("'card_y' is only available for classification datasets.")
        return card_y

    @property
    @final
    def target_dim(self) -> int:
        self._check_setup_called()
        card_y = self.train_data.n_classes
        if (card_y := self.train_data.n_classes) is None:
            return self.dim_y
        return card_y

    @final
    def _dataloader(
        self,
        ds: Union[LabeledDataset, UnlabeledDataset],
        *,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        batch_sampler: Optional[Sampler[Sequence[int]]] = None,
        sampler: Optional[Sampler[int]] = None,
    ) -> DataLoader:
        """Factory method for all dataloaders."""
        converter = None if isinstance(ds, DatasetWrapper) else sample_converter
        return CdtDataLoader(
            ds,  # type: ignore
            batch_size=batch_size if batch_sampler is None else 1,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            persistent_workers=self.persist_workers,
            batch_sampler=batch_sampler,
            sampler=sampler,
            converter=converter,
            cast_to_sample=False,
        )

    @final
    def _train_dataloader(
        self,
        ds: Union[LabeledDataset, UnlabeledDataset],
        *,
        batch_size: Optional[int] = None,
    ) -> DataLoader:
        """Factory method for train-data dataloaders."""
        base_dataset = ds.dataset if isinstance(ds, DatasetWrapper) else ds
        if batch_size is None:
            batch_size = self.eval_batch_size if eval else self.train_batch_size_l
        batch_sampler = None

        if self.sampling_method is not SamplingMethod.standard:
            sampling_groups = (
                self.groupby_fields if self.sampling_groups is None else self.sampling_groups
            )
            grouper = CombinatorialGrouper(
                dataset=base_dataset, groupby_fields=self.sampling_groups
            )
            group_ids, group_counts = grouper.metadata_to_group(
                base_dataset.metadata_array, return_counts=True
            )

            if self.sampling_method is SamplingMethod.stratified:
                num_groups = len(group_ids.unique())
                num_samples_per_group = batch_size // num_groups
                if batch_size % num_groups:
                    self.logger.info(
                        f"For stratified sampling, the batch size must be a multiple of the number of groups."
                        f"Since the batch size is not integer divisible by the number of groups ({num_groups}),"
                        f"the batch size is being reduced to {num_samples_per_group * num_groups}."
                    )

                batch_sampler = StratifiedBatchSampler(
                    group_ids=group_ids.squeeze().tolist(),
                    num_samples_per_group=num_samples_per_group,
                    shuffle=True,
                    base_sampler="sequential",
                    training_mode=self.training_mode,
                    drop_last=False,
                )

            elif self.sampling_method is SamplingMethod.weighted:
                group_counts_r = group_counts.reciprocal()
                group_weights = group_counts_r / group_counts_r.sum()
                weights = group_weights[group_ids]
                batch_sampler = WeightedBatchSampler(
                    weights=weights.squeeze(),
                    batch_size=batch_size,
                    replacement=True,
                )
            else:
                batch_sampler = HierarchicalSampler(
                    group_ids=group_ids,
                    batch_size=batch_size,
                    uniform=True,
                )
        else:
            batch_sampler = SequentialBatchSampler(
                data_source=ds,
                batch_size=batch_size,
                shuffle=True,
                training_mode=self.training_mode,
                drop_last=False,
            )

        return self._dataloader(batch_size=batch_size, ds=ds, batch_sampler=batch_sampler)

    @implements(pl.LightningDataModule)
    def train_dataloader(self, *, eval: bool = False) -> Union[DataLoader, LabeledUnlabeledDlPair]:
        if eval:
            ds_l = gcopy(self.train_data.labeled, deep=False)
            ds_l.transform = self.test_transforms  # type: ignore
            dl_labeled = self._dataloader(
                ds=ds_l, batch_size=self.eval_batch_size, shuffle=False, drop_last=False
            )
        else:
            dl_labeled = self._train_dataloader(
                ds=self.train_data.labeled,
                batch_size=self.train_batch_size_l,
            )
        if self.train_data.unlabeled is None:
            return dl_labeled

        if eval:
            ds_u = gcopy(self.train_data.unlabeled, deep=False)
            ds_u.transform = self.test_transforms  # type: ignore
            dl_u = self._dataloader(
                ds=ds_u, batch_size=self.eval_batch_size, shuffle=False, drop_last=False
            )
        else:
            dl_u = self._train_dataloader(
                ds=self.train_data.unlabeled,
                batch_size=self.train_batch_size_u,
            )
        return {"labeled": dl_labeled, "unlabeled": dl_u}

    @implements(pl.LightningDataModule)
    def val_dataloader(self) -> CombinedLoader:
        val_loaders = {
            f"{str(Stage.validate)}/{name}": self._dataloader(
                ds=split, batch_size=self.eval_batch_size
            )
            for name, split in self.val_data.labeled.items()
        }
        test_loaders = {
            f"{str(Stage.test)}/{name}": self._dataloader(ds=split, batch_size=self.eval_batch_size)
            for name, split in self.test_data.labeled.items()
        }
        return CombinedLoader(
            val_loaders | test_loaders,
            mode="max_size_cycle",
        )

    @implements(pl.LightningDataModule)
    def test_dataloader(self) -> CombinedLoader:
        return CombinedLoader(
            {
                f"{str(Stage.test)}/{name}": self._dataloader(
                    ds=split, batch_size=self.eval_batch_size
                )
                for name, split in self.test_data.labeled.items()
            },
            mode="max_size_cycle",
        )

    @abstractmethod
    def _get_splits(self) -> TrainValTestSplit:
        ...

    @implements(pl.LightningDataModule)
    @final
    def setup(self, stage: Optional[Stage] = None, force_reset: bool = False) -> None:
        # Only perform the setup if it hasn't already been done
        if force_reset or (not self.is_set_up):
            self._setup(stage=stage)

    def _setup(self, stage: Optional[Stage] = None) -> None:
        splits = self._get_splits()
        self._train_data = splits.train
        self._val_data = splits.val
        self._test_data = splits.test


@attr.define(kw_only=True)
class WILDSVisionDataModule(WILDSDataModule):

    root: Path = attr.field(kw_only=False)
    _train_transforms_l: Optional[ImageToTensorTransform] = None
    _train_transforms_u: Optional[ImageToTensorTransform] = None
    _test_transforms: Optional[ImageToTensorTransform] = None

    @property
    @final
    def train_transforms_l(self) -> ImageToTensorTransform:
        return (
            self._default_train_transforms
            if self._train_transforms_l is None
            else self._train_transforms_l
        )

    @train_transforms_l.setter
    def train_transforms_l(self, transform: Optional[ImageToTensorTransform]) -> None:  # type: ignore
        self._train_transforms_l = transform
        if self._train_data is not None:
            self._train_data.labeled.transform = transform

    @property
    @final
    def train_transforms_u(self) -> ImageToTensorTransform:
        return (
            self.train_transforms_l
            if self._train_transforms_u is None
            else self._train_transforms_u
        )

    @train_transforms_u.setter
    def train_transforms_u(self, transform: Optional[ImageToTensorTransform]) -> None:  # type: ignore
        self._train_transforms_u = transform
        if (self._train_data is not None) and (self._train_data.unlabeled is not None):
            self._train_data.unlabeled.transform = transform

    @property
    @final
    def test_transforms(self) -> ImageToTensorTransform:
        return (
            self._default_test_transforms
            if self._test_transforms is None
            else self._test_transforms
        )

    @test_transforms.setter
    @final
    def test_transforms(self, transform: Optional[ImageToTensorTransform]) -> None:  # type: ignore
        self._test_transforms = transform
        if self._test_data is not None:
            for subset in self._test_data:
                subset.transform = transform
        if self._val_data is not None:
            for subset in self._val_data:
                subset.transform = transform

    @property
    @abstractmethod
    def _default_train_transforms(self) -> ImageToTensorTransform:
        ...

    @property
    @abstractmethod
    def _default_test_transforms(self) -> ImageToTensorTransform:
        ...

    @implements(WILDSDataModule)
    @final
    def _setup(self, stage: Optional[Stage] = None) -> None:
        splits = self._get_splits()
        self._train_data = splits.train
        self._val_data = splits.val
        self._test_data = splits.test

        self.train_transforms_l = self.train_transforms_l
        self.train_transforms_u = self.train_transforms_u
        self.test_transforms = self.test_transforms
        self._evaluator = self.test_data.ood.labeled.eval


BatchEncoder: TypeAlias = Callable[[str], DictContainer]


@attr.define(kw_only=True)
class WILDSTextDataModule(WILDSDataModule):

    root: Path = attr.field(kw_only=False)
    max_token_length: int = 512
    _train_transforms_l: Optional[PreTrainedTokenizerBase] = None
    _train_transforms_u: Optional[PreTrainedTokenizerBase] = None
    _test_transforms: Optional[PreTrainedTokenizerBase] = None

    @overload
    def _partial(self, transform: PreTrainedTokenizerBase) -> BatchEncoder:
        ...

    @overload
    def _partial(self, transform: None) -> None:
        ...

    def _partial(self, transform: Optional[PreTrainedTokenizerBase]) -> Optional[BatchEncoder]:
        if transform is not None:

            def _closure(x: str) -> DictContainer[Tensor]:
                out = DictContainer(
                    transform(
                        x,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_token_length,
                        return_tensors="pt",
                    )
                )
                return out

            return _closure

    @property
    @final
    def train_transforms_l(self) -> PreTrainedTokenizerBase:
        return (
            self._default_train_transforms
            if self._train_transforms_l is None
            else self._train_transforms_l
        )

    @train_transforms_l.setter
    def train_transforms_l(self, transform: Optional[PreTrainedTokenizerBase]) -> None:  # type: ignore
        self._train_transforms_l = transform
        if self._train_data is not None:
            partialled = self._partial(transform)
            self._train_data.labeled.transform = partialled

    @property
    @final
    def train_transforms_u(self) -> PreTrainedTokenizerBase:
        return (
            self.train_transforms_l
            if self._train_transforms_u is None
            else self._train_transforms_u
        )

    @train_transforms_u.setter
    def train_transforms_u(self, transform: Optional[PreTrainedTokenizerBase]) -> None:  # type: ignore
        self._train_transforms_u = transform
        if (self._train_data is not None) and (self._train_data.unlabeled is not None):
            partialled = self._partial(transform)
            self._train_data.unlabeled.transform = partialled

    @property
    @final
    def test_transforms(self) -> PreTrainedTokenizerBase:
        return (
            self._default_test_transforms
            if self._test_transforms is None
            else self._test_transforms
        )

    @test_transforms.setter
    @final
    def test_transforms(self, transform: PreTrainedTokenizerBase) -> None:  # type: ignore
        self._test_transforms = transform
        partialled = self._partial(transform)
        if self._test_data is not None:
            for subset in self._test_data:
                subset.transform = partialled
        if self._val_data is not None:
            for subset in self._val_data:
                subset.transform = partialled

    @property
    @abstractmethod
    def _default_train_transforms(self) -> PreTrainedTokenizerBase:
        return BertTokenizerFast.from_pretrained("bert-base-uncased")

    @property
    @abstractmethod
    def _default_test_transforms(self) -> PreTrainedTokenizerBase:
        return BertTokenizerFast.from_pretrained("bert-base-uncased")

    @implements(WILDSDataModule)
    @final
    def _setup(self, stage: Optional[Stage] = None) -> None:
        splits = self._get_splits()
        self._train_data = splits.train
        self._val_data = splits.val
        self._test_data = splits.test

        self.train_transforms_l = self.train_transforms_l
        self.train_transforms_u = self.train_transforms_u
        self.test_transforms = self.test_transforms
        self._evaluator = self.test_data.ood.labeled.eval

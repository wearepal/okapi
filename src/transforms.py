from abc import abstractmethod
from dataclasses import dataclass
import random
from typing import (
    Callable,
    Final,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from conduit.data.constants import IMAGENET_STATS
from conduit.data.datasets.utils import (
    ImageTform,
    PillowTform,
    apply_image_transform,
    img_to_tensor,
)
from conduit.data.structures import (
    InputContainer,
    MeanStd,
    RawImage,
    concatenate_inputs,
)
import numpy as np
from ranzen.decorators import implements
from ranzen.misc import gcopy
import torch
from torch import Tensor
import torchvision.transforms as T  # type: ignore
import torchvision.transforms.functional as TF  # type: ignore
from typing_extensions import Protocol, Self

__class__ = [
    "BatchTransform",
    "FIX_MATCH_AUGMENTATION_POOL",
    "FixMatchPair",
    "FixMatchRandAugment",
    "FixMatchTransform",
    "Identity",
    "ImageToTensorTransform",
    "TextToTensorTransform",
    "MultiCropOutput",
    "RandAugmentPM",
    "RandomGaussianBlur",
    "RandomSolarize",
]


I = TypeVar("I", bound=RawImage)


class Identity(Generic[I]):
    def __call__(self, image: I) -> I:
        return image


T_co = TypeVar("T_co", bound=Union[Tensor, InputContainer[Tensor]], covariant=True)


class ImageToTensorTransform(Protocol[T_co]):
    def __call__(self, image: Image.Image) -> T_co:
        ...


class BatchTransform(Protocol):
    @overload
    def __call__(self, inputs: Tensor, *, targets: Tensor) -> Tuple[Tensor, Tensor]:
        ...

    @overload
    def __call__(self, inputs: Tensor, *, targets: None = ...) -> Tensor:
        ...

    def __call__(
        self, inputs: Tensor, *, targets: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        ...


X = TypeVar("X", bound=Union[Tensor, RawImage, List[Image.Image]])


@dataclass
class FixMatchPair(InputContainer[X]):
    strong: X
    weak: X

    @implements(InputContainer)
    def __len__(self) -> int:
        if isinstance(self.strong, Image.Image):
            return 1
        return len(self.strong)

    @implements(InputContainer)
    def __add__(self, other: Self) -> Self:
        copy = gcopy(self, deep=False)
        is_batched = isinstance(self.strong, (Tensor, np.ndarray)) and (self.strong.ndim == 4)
        copy.strong = concatenate_inputs(copy.strong, other.strong, is_batched=is_batched)
        copy.weak = concatenate_inputs(copy.weak, other.weak, is_batched=is_batched)

        return copy


A = TypeVar("A", bound=ImageTform)


class FixMatchTransform(Generic[A]):
    def __init__(
        self,
        strong_transform: A,
        *,
        weak_transform: A,
        shared_transform_start: Optional[Callable[[Union[RawImage, Tensor]], RawImage]] = None,
        shared_transform_end: Optional[
            Callable[[Union[RawImage, Tensor]], Union[RawImage, Tensor]]
        ] = None,
    ) -> None:
        self.strong_transform = strong_transform
        self.weak_transform = weak_transform
        self.shared_transform_start = shared_transform_start
        self.shared_transform_end = shared_transform_end

    def __call__(self, image: RawImage) -> FixMatchPair:
        if self.shared_transform_start is not None:
            image = self.shared_transform_start(image)

        strongly_augmented_image = apply_image_transform(
            image=image, transform=self.strong_transform
        )

        weakly_augmented_image = apply_image_transform(image=image, transform=self.weak_transform)
        if self.shared_transform_end is not None:
            strongly_augmented_image = self.shared_transform_end(strongly_augmented_image)
            weakly_augmented_image = self.shared_transform_end(weakly_augmented_image)

        return FixMatchPair(strong=strongly_augmented_image, weak=weakly_augmented_image)


def _sample_uniform(a: float, b: float) -> float:
    return torch.empty(1).uniform_(a, b).item()


class RandAugmentOp:
    @staticmethod
    @abstractmethod
    def __call__(img: Image.Image, v: float) -> Image.Image:
        ...


class AutoContrastRA(RandAugmentOp):
    @staticmethod
    def __call__(img: Image.Image, v: float) -> Image.Image:
        return ImageOps.autocontrast(img)


class BrightnessRA(RandAugmentOp):
    @staticmethod
    def __call__(img: Image.Image, v: float) -> Image.Image:
        assert v >= 0.0
        return ImageEnhance.Brightness(img).enhance(v)


class ColorRA(RandAugmentOp):
    @staticmethod
    def __call__(img: Image.Image, v: float) -> Image.Image:
        assert v >= 0.0
        return ImageEnhance.Color(img).enhance(v)


class ContrastRA(RandAugmentOp):
    @staticmethod
    def __call__(img: Image.Image, v: float) -> Image.Image:
        assert v >= 0.0
        return ImageEnhance.Contrast(img).enhance(v)


class EqualizeRA(RandAugmentOp):
    @staticmethod
    def __call__(img: Image.Image, v: float) -> Image.Image:
        return ImageOps.equalize(img)


class InvertRA(RandAugmentOp):
    @staticmethod
    def __call__(img: Image.Image, v: float) -> Image.Image:
        return ImageOps.invert(img)


class IdentityRA(RandAugmentOp):
    @staticmethod
    def __call__(img: Image.Image, v: float) -> Image.Image:
        return img


class PosterizeRA(RandAugmentOp):  # [4, 8]
    @staticmethod
    def __call__(img: Image.Image, v: float) -> Image.Image:
        v = int(v)
        v = max(1, v)
        return ImageOps.posterize(img, v)


class RotateRA(RandAugmentOp):  # [-30, 30]
    @staticmethod
    def __call__(img: Image.Image, v: float) -> Image.Image:
        return img.rotate(v)


class SharpnessRA(RandAugmentOp):  # [0.1,1.9]
    @staticmethod
    def __call__(img: Image.Image, v: float) -> Image.Image:
        assert v >= 0.0
        return ImageEnhance.Sharpness(img).enhance(v)


class ShearXRA(RandAugmentOp):  # [-0.3, 0.3]
    @staticmethod
    def __call__(img: Image.Image, v: float) -> Image.Image:
        return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))


class ShearYRA(RandAugmentOp):  # [-0.3, 0.3]
    @staticmethod
    def __call__(img: Image.Image, v: float) -> Image.Image:
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))


class TranslateXRA(RandAugmentOp):  # [-150, 150] => percentage: [-0.45, 0.45]
    @staticmethod
    def __call__(img: Image.Image, v: float) -> Image.Image:
        v = v * img.size[0]
        return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


class TranslateXabsRA(RandAugmentOp):  # [-150, 150] => percentage: [-0.45, 0.45]
    @staticmethod
    def __call__(img: Image.Image, v: float) -> Image.Image:
        return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


class TranslateYRA(RandAugmentOp):  # [-150, 150] => percentage: [-0.45, 0.45]
    @staticmethod
    def __call__(img: Image.Image, v: float) -> Image.Image:
        v = v * img.size[1]
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


class TranslateYabsRA(RandAugmentOp):  # [-150, 150] => percentage: [-0.45, 0.45]
    @staticmethod
    def __call__(img: Image.Image, v: float) -> Image.Image:
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


class SolarizeRA(RandAugmentOp):  # [0, 256]
    @staticmethod
    def __call__(img: Image.Image, v: int) -> Image.Image:
        assert 0 <= v <= 256
        return ImageOps.solarize(img, v)


class CutOutRA(RandAugmentOp):  # [0, 60] => percentage: [0, 0.2] => change to [0, 0.5]
    @staticmethod
    def __call__(img: Image.Image, v: float) -> Image.Image:
        assert 0.0 <= v <= 0.5

        v = v * img.size[0]
        return CutoutAbsRA()(img, v)


class CutoutAbsRA(RandAugmentOp):  # [0, 60] => percentage: [0, 0.2]
    @staticmethod
    def __call__(img: Image.Image, v: float) -> Image.Image:
        if v < 0:
            return img
        w, h = img.size
        x_center = _sample_uniform(0, w)
        y_center = _sample_uniform(0, h)

        x0 = int(max(0, x_center - v / 2.0))
        y0 = int(max(0, y_center - v / 2.0))
        x1 = min(w, x0 + v)
        y1 = min(h, y0 + v)

        xy = (x0, y0, x1, y1)
        color = (125, 123, 114)
        img = img.copy()
        ImageDraw.Draw(img).rectangle(xy, color)
        return img


FIX_MATCH_AUGMENTATION_POOL: Final[Tuple[Tuple[Type[RandAugmentOp], float, float], ...]] = (
    (AutoContrastRA, 0, 1),
    (BrightnessRA, 0.05, 0.95),
    (ColorRA, 0.05, 0.95),
    (ContrastRA, 0.05, 0.95),
    (EqualizeRA, 0, 1),
    (IdentityRA, 0, 1),
    (PosterizeRA, 4, 8),
    (RotateRA, -30, 30),
    (SharpnessRA, 0.05, 0.95),
    (ShearXRA, -0.3, 0.3),
    (ShearYRA, -0.3, 0.3),
    (SolarizeRA, 0, 256),
    (TranslateXRA, -0.3, 0.3),
    (TranslateYRA, -0.3, 0.3),
)


class FixMatchRandAugment:
    def __init__(
        self,
        num_ops: int,
        augmentation_pool: Sequence[
            Tuple[Type[RandAugmentOp], float, float]
        ] = FIX_MATCH_AUGMENTATION_POOL,
    ) -> None:
        assert num_ops >= 1, "RandAugment N has to be a value greater than or equal to 1."
        self.n = num_ops
        self.augmentation_pool = augmentation_pool

    def __call__(self, img: Image.Image) -> Image.Image:
        ops = [
            self.augmentation_pool[torch.randint(len(self.augmentation_pool), (1,))]
            for _ in range(self.n)
        ]
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * _sample_uniform(0, 1)
            img = op()(img=img, v=val)
        cutout_val = _sample_uniform(0, 1) * 0.5
        return CutOutRA()(img=img, v=cutout_val)


class RandomGaussianBlur:
    """
    Apply Gaussian Blur to the PIL image with some probability.
    """

    def __init__(self, p: float = 0.5, *, radius_min: float = 0.1, radius_max: float = 2.0) -> None:
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img: Image.Image) -> Image.Image:
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max))
        )


class RandomSolarize:
    """
    Apply Solarization to a PIL image with some probability.
    """

    def __init__(self, p: float) -> None:
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img


@dataclass
class MultiViewPair(InputContainer[Tensor]):
    v1: Tensor
    v2: Tensor

    def __post_init__(self) -> None:
        if self.v1.size() != self.v2.size():
            raise AttributeError("'v1' and 'v2' must have the same shape.")

    @implements(InputContainer)
    def __len__(self) -> int:
        return len(self.v1)

    @implements(InputContainer)
    def __add__(self, other: Self) -> Self:
        copy = gcopy(self, deep=False)
        is_batched = self.v1.ndim == 4
        copy.v1 = concatenate_inputs(copy.v1, other.v1, is_batched=is_batched)
        copy.v2 = concatenate_inputs(copy.v2, other.v2, is_batched=is_batched)

        return copy

    def size(self) -> torch.Size:
        return self.v1.size()

    def shape(self) -> torch.Size:
        return self.v1.shape

    def merge(self) -> Tensor:
        return torch.cat((self.v1, self.v2), dim=0)

    @property
    def anchor(self) -> Tensor:
        return self.v1

    @property
    def target(self) -> Tensor:
        return self.v2

    @property
    def num_sources(self) -> int:
        return len(self)


@dataclass
class MultiCropOutput(InputContainer[MultiViewPair]):
    global_views: MultiViewPair
    local_views: Tensor

    @property
    def num_sources(self) -> int:
        """The number of samples from which the views were generated."""
        return len(self.global_views)

    @property
    def num_global_crops(self) -> int:
        return 2

    @property
    def num_local_crops(self) -> int:
        if self.local_views is None:
            return 0
        return len(self.local_views) // len(self.global_views)

    @property
    def num_crops(self) -> int:
        return self.num_global_crops + self.num_local_crops

    @property
    def global_crop_size(self) -> Tuple[int, int, int]:
        return self.global_views.shape[1:]  # type: ignore

    @property
    def local_crop_size(self) -> Tuple[int, int, int]:
        if self.local_views is None:
            raise AttributeError("Cannot retrieve the local-crop size as 'local_' is 'None'.")
        return self.local_views.shape[1:]

    @property
    def shape(self):
        """Shape of the global crops."""
        return self.global_views.shape

    def astuple(self) -> Tuple[Tensor, Tensor]:
        return (self.global_views.merge(), self.local_views)

    @property
    def anchor(self) -> Tuple[Tensor, Tensor]:
        return (self.global_views.v1, self.local_views)

    @property
    def target(self) -> Tensor:
        return self.global_views.v2

    @implements(InputContainer)
    def __len__(self) -> int:
        """Total number of crops."""
        return len(self.global_views) + len(self.local_views)

    @implements(InputContainer)
    def __add__(self, other: Self) -> Self:
        copy = gcopy(self, deep=False)
        copy.global_views = copy.global_views + other.global_views
        if copy.local_views is None:
            if other.local_views is not None:
                copy.local_views = other.local_views
        else:
            if other.local_views is not None:
                copy.local_views = copy.local_views + other.local_views
                is_batched = copy.local_views.ndim == 4
                copy.local_views = concatenate_inputs(
                    copy.local_views, other.local_views, is_batched=is_batched
                )
        return copy


LT = TypeVar("LT", bound=Optional[ImageTform])


class MultiCropTransform(Generic[LT]):
    def __init__(
        self,
        *,
        global_transform_1: ImageTform,
        global_transform_2: Optional[ImageTform] = None,
        local_transform: LT = None,
        local_crops_number: int = 8,
    ) -> None:
        self.global_transform_1 = global_transform_1
        self.global_transform_2 = (
            global_transform_1 if global_transform_2 is None else global_transform_2
        )
        if (local_transform is not None) and (local_crops_number <= 0):
            raise AttributeError(
                " 'local_crops' must be a positive integer if 'local_transform' is defined."
            )
        self.local_transform = local_transform
        self.local_crops_number = local_crops_number

    @staticmethod
    def _apply_transform(image: RawImage, transform: ImageTform):
        view = apply_image_transform(image, transform=transform)
        if not isinstance(view, Tensor):
            view = img_to_tensor(view)
        return view

    @overload
    def __call__(self: "MultiCropTransform[ImageTform]", image: RawImage) -> MultiCropOutput:
        ...

    @overload
    def __call__(self: "MultiCropTransform[None]", image: RawImage) -> MultiViewPair:
        ...

    def __call__(
        self: "MultiCropTransform", image: RawImage
    ) -> Union[MultiCropOutput, MultiViewPair]:
        global_crop_v1 = self._apply_transform(image, transform=self.global_transform_1)
        global_crop_v2 = self._apply_transform(image, transform=self.global_transform_2)
        gc_pair = MultiViewPair(v1=global_crop_v1, v2=global_crop_v2)

        if (self.local_transform is None) or (self.local_crops_number == 0):
            return gc_pair
        local_crops = torch.stack(
            [
                self._apply_transform(image, transform=self.local_transform)
                for _ in range(self.local_crops_number)
            ],
            dim=0,
        )

        return MultiCropOutput(global_views=gc_pair, local_views=local_crops)

    @classmethod
    def with_dino_transform(
        cls,
        *,
        global_crop_size: Union[int, Sequence[int]] = 224,
        local_crop_size: Union[int, Sequence[int]] = 96,
        norm_values: Optional[MeanStd] = IMAGENET_STATS,
        global_crops_scale: Tuple[float, float] = (0.4, 1.0),
        local_crops_scale: Tuple[float, float] = (0.05, 0.4),
        local_crops_number: int = 8,
    ) -> "MultiCropTransform":

        flip_and_color_jitter = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(
                    [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
                ),
                T.RandomGrayscale(p=0.2),
            ]
        )
        normalize_ls: List[PillowTform] = [T.ToTensor()]
        if norm_values is not None:
            normalize_ls.append(
                T.Normalize(mean=norm_values.mean, std=norm_values.std),
            )
        normalize = T.Compose(normalize_ls)

        # first global crop
        global_transform_1 = T.Compose(
            [
                T.RandomResizedCrop(
                    global_crop_size,
                    scale=global_crops_scale,
                    interpolation=TF.InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                RandomGaussianBlur(1.0),
                normalize,
            ]
        )
        # second global crop
        global_transform_2 = T.Compose(
            [
                T.RandomResizedCrop(
                    global_crop_size,
                    scale=global_crops_scale,
                    interpolation=TF.InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                RandomGaussianBlur(0.1),
                RandomSolarize(0.2),
                normalize,
            ]
        )
        # transformation for the local small crops
        local_transform = None
        if local_crops_number > 0:
            local_transform = T.Compose(
                [
                    T.RandomResizedCrop(
                        local_crop_size,
                        scale=local_crops_scale,
                        interpolation=TF.InterpolationMode.BICUBIC,
                    ),
                    flip_and_color_jitter,
                    RandomGaussianBlur(p=0.5),
                    normalize,
                ]
            )

        return MultiCropTransform(
            global_transform_1=global_transform_1,
            global_transform_2=global_transform_2,
            local_transform=local_transform,
            local_crops_number=local_crops_number,
        )


def rgb_transform_pm(ms_img: Tensor, transform: PillowTform) -> Tensor:
    from wilds.datasets.poverty_dataset import (  # type: ignore
        _MEANS_2009_17,
        _STD_DEVS_2009_17,
    )

    poverty_rgb_means = np.array([_MEANS_2009_17[c] for c in ["RED", "GREEN", "BLUE"]]).reshape(
        (-1, 1, 1)
    )
    poverty_rgb_stds = np.array([_STD_DEVS_2009_17[c] for c in ["RED", "GREEN", "BLUE"]]).reshape(
        (-1, 1, 1)
    )

    def unnormalize_rgb_in_poverty_ms_img(ms_img: Tensor) -> Tensor:
        result = ms_img.detach().clone()
        result[:3] = (result[:3] * poverty_rgb_stds) + poverty_rgb_means
        return result

    def normalize_rgb_in_poverty_ms_img(ms_img: Tensor) -> Tensor:
        result = ms_img.detach().clone()
        result[:3] = (result[:3] - poverty_rgb_means) / poverty_rgb_stds
        return ms_img

    color_transform = T.Compose(
        [
            T.Lambda(lambda ms_img: unnormalize_rgb_in_poverty_ms_img(ms_img)),
            transform,
            T.Lambda(lambda ms_img: normalize_rgb_in_poverty_ms_img(ms_img)),
        ]
    )
    # The first three channels of the Poverty MS images are BGR
    # So we shuffle them to the standard RGB to do the ColorJitter
    # Before shuffling them back
    ms_img[:3] = color_transform(ms_img[[2, 1, 0]])[[2, 1, 0]]  # bgr to rgb to bgr
    return ms_img


class RandAugmentPM:
    def __init__(self) -> None:
        self._transform = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=0.1, scale=(0.9, 1.1)),
                T.Lambda(lambda ms_img: self._color_jitter(ms_img)),
                T.Lambda(lambda ms_img: self._cutout(ms_img)),
            ]
        )

    @staticmethod
    def _color_jitter(ms_img: Tensor):
        return rgb_transform_pm(
            ms_img, T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.1)
        )

    @staticmethod
    def _cutout(ms_img: Tensor):
        def _sample_uniform(a, b):
            return torch.empty(1).uniform_(a, b).item()

        assert ms_img.shape[1] == ms_img.shape[2]
        img_width = ms_img.shape[1]
        cutout_width = _sample_uniform(0, img_width / 2)
        cutout_center_x = _sample_uniform(0, img_width)
        cutout_center_y = _sample_uniform(0, img_width)
        x0 = int(max(0, cutout_center_x - cutout_width / 2))
        y0 = int(max(0, cutout_center_y - cutout_width / 2))
        x1 = int(min(img_width, cutout_center_x + cutout_width / 2))
        y1 = int(min(img_width, cutout_center_y + cutout_width / 2))

        # Fill with 0 because the data is already normalized to mean zero
        ms_img[:, x0:x1, y0:y1] = 0
        return ms_img

    def __call__(self, image: Tensor) -> Tensor:
        return self._transform(image)

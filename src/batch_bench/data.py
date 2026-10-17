"""Image sources, preprocessing, and DataLoader construction.

The benchmark feeds the GPU through a real :class:`~torch.utils.data.DataLoader`
so that JPEG decoding and HuggingFace preprocessing compete for CPU time
exactly as they would in production. Measuring the model alone would
overstate the useful batch size whenever the input pipeline is the bottleneck.
"""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# (width, height), matching PIL's convention.
ImageSize = tuple[int, int]

DEFAULT_IMAGE_SIZE: ImageSize = (640, 480)
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


class PreprocessMode(StrEnum):
    """Where the HuggingFace image processor runs inside the DataLoader workers."""

    # The processor sees the whole batch in ``collate_fn``; fast (torchvision)
    # processors vectorize this and pad mixed sizes themselves.
    BATCH = "batch"
    # The processor runs once per image in ``__getitem__``, the classic pattern.
    SAMPLE = "sample"


def parse_image_size(text: str) -> ImageSize:
    """Parse ``WIDTHxHEIGHT`` (e.g. ``640x480``)."""
    parts = text.lower().split("x")
    if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
        raise ValueError(f"image size must look like 640x480, got {text!r}")
    width, height = (int(part) for part in parts)
    if width < 1 or height < 1:
        raise ValueError(f"image size must be positive, got {text!r}")
    return width, height


class ImageSource(Protocol):
    """A finite, indexable collection of RGB images."""

    def __len__(self) -> int: ...

    def load(self, index: int) -> Image.Image:
        """Decode image ``index``."""
        ...


class ImageProcessor(Protocol):
    """The part of a HuggingFace image processor the benchmark uses."""

    def __call__(self, images: Image.Image | list[Image.Image], return_tensors: str) -> Mapping[str, Any]:
        """Preprocess one image or a list of images into model inputs."""
        ...


def _smooth_random_image(rng: np.random.Generator, size: ImageSize) -> Image.Image:
    """Upsampled low-resolution noise: compresses and decodes like a photo, unlike white noise."""
    width, height = size
    coarse = rng.integers(0, 256, size=(max(1, height // 16), max(1, width // 16), 3), dtype=np.uint8)
    return Image.fromarray(coarse).resize(size, Image.Resampling.BICUBIC)


class SyntheticImageSource:
    """A pool of random JPEGs that are decoded on every access.

    Keeping JPEG bytes rather than decoded pixels makes each sample pay a
    realistic decode cost without needing a dataset on disk.

    Args:
        size: Image size as ``(width, height)``.
        pool_size: Number of distinct images to cycle through.
        seed: Seed for reproducible pixels.
        quality: JPEG quality.
    """

    def __init__(
        self,
        size: ImageSize = DEFAULT_IMAGE_SIZE,
        pool_size: int = 32,
        seed: int = 0,
        quality: int = 90,
    ) -> None:
        if pool_size < 1:
            raise ValueError("pool_size must be at least 1")
        self.size = size
        rng = np.random.default_rng(seed)
        self._jpegs: list[bytes] = []
        for _ in range(pool_size):
            buffer = io.BytesIO()
            _smooth_random_image(rng, size).save(buffer, format="JPEG", quality=quality)
            self._jpegs.append(buffer.getvalue())

    def __len__(self) -> int:
        return len(self._jpegs)

    def load(self, index: int) -> Image.Image:
        """Decode image ``index`` (modulo the pool size)."""
        with Image.open(io.BytesIO(self._jpegs[index % len(self._jpegs)])) as image:
            return image.convert("RGB")


class FolderImageSource:
    """Images found recursively under ``root``.

    Args:
        root: Directory to search.
        resize: Optional ``(width, height)`` every image is resized to, which
            makes per-sample preprocessing produce equal-sized tensors.

    Raises:
        FileNotFoundError: If ``root`` holds no supported images.
    """

    def __init__(self, root: Path, resize: ImageSize | None = None) -> None:
        self.root = root
        self.resize = resize
        self.files: list[Path] = sorted(
            path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.files:
            extensions = ", ".join(sorted(IMAGE_EXTENSIONS))
            raise FileNotFoundError(f"no images ({extensions}) found under {root}")

    def __len__(self) -> int:
        return len(self.files)

    def load(self, index: int) -> Image.Image:
        """Decode file ``index`` (modulo the file count)."""
        with Image.open(self.files[index % len(self.files)]) as image:
            rgb = image.convert("RGB")
        if self.resize is not None and rgb.size != self.resize:
            rgb = rgb.resize(self.resize, Image.Resampling.BILINEAR)
        return rgb


class ImageDataset(Dataset[Image.Image]):
    """``length`` images drawn by cycling over ``source``.

    Sources are usually smaller than ``batch_size * batches``; cycling lets any
    batch size run for as many batches as the benchmark needs.
    """

    def __init__(self, source: ImageSource, length: int) -> None:
        self.source = source
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Image.Image:
        if not 0 <= index < self.length:
            raise IndexError(index)
        return self.source.load(index)


def _tensors_only(features: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in features.items() if isinstance(value, torch.Tensor)}


class PreprocessedDataset(Dataset[dict[str, torch.Tensor]]):
    """Runs the image processor on each image individually (``--preprocess sample``)."""

    def __init__(self, images: ImageDataset, processor: ImageProcessor) -> None:
        self.images = images
        self.processor = processor

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        features = self.processor(images=self.images[index], return_tensors="pt")
        # The processor returns a batch of one; drop that leading dimension.
        return {key: value[0] for key, value in _tensors_only(features).items()}


class BatchProcessorCollator:
    """``collate_fn`` that preprocesses a whole batch at once (``--preprocess batch``)."""

    def __init__(self, processor: ImageProcessor) -> None:
        self.processor = processor

    def __call__(self, images: list[Image.Image]) -> dict[str, torch.Tensor]:
        """Preprocess ``images`` into one batch of tensors."""
        return _tensors_only(self.processor(images=images, return_tensors="pt"))


def pad_collate(samples: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack per-sample tensors, zero-padding trailing dimensions when shapes differ."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch: dict[str, torch.Tensor] = {}
    for key in samples[0]:
        tensors = [sample[key] for sample in samples]
        first = tensors[0]
        if all(tensor.shape == first.shape for tensor in tensors):
            batch[key] = torch.stack(tensors)
            continue
        if any(tensor.dim() != first.dim() for tensor in tensors):
            raise ValueError(f"cannot pad {key!r}: samples have different ranks")
        max_shape = [max(tensor.shape[dim] for tensor in tensors) for dim in range(first.dim())]
        padded = first.new_zeros((len(tensors), *max_shape))
        for row, tensor in enumerate(tensors):
            padded[(row, *(slice(0, size) for size in tensor.shape))] = tensor
        batch[key] = padded
    return batch


def build_loader(
    images: ImageDataset,
    processor: ImageProcessor,
    *,
    batch_size: int,
    num_workers: int,
    preprocess: PreprocessMode = PreprocessMode.BATCH,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
    mp_context: str | None = None,
) -> DataLoader[Any]:
    """Build the DataLoader for one trial.

    ``drop_last`` keeps every measured batch at exactly ``batch_size``.
    ``mp_context`` picks how workers start (``fork``, ``spawn`` or
    ``forkserver``); None keeps the platform default.
    """
    dataset: Dataset[Any]
    if preprocess is PreprocessMode.SAMPLE:
        dataset, collate = PreprocessedDataset(images, processor), pad_collate
    else:
        dataset, collate = images, BatchProcessorCollator(processor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_memory,
        drop_last=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        multiprocessing_context=mp_context if num_workers > 0 else None,
    )

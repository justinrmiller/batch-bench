"""Image sources, preprocessing, and DataLoader construction.

The benchmark feeds the GPU through a real :class:`~torch.utils.data.DataLoader`
so that JPEG decoding and HuggingFace preprocessing compete for CPU time
exactly as they would in production. Measuring the model alone would
overstate the useful batch size whenever the input pipeline is the bottleneck.
"""

from __future__ import annotations

import io
import math
import random
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, cast, overload

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, get_worker_info

# (width, height), matching PIL's convention.
ImageSize = tuple[int, int]

DEFAULT_IMAGE_SIZE: ImageSize = (640, 480)
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})

# Images per image-processor call with ``--preprocess batch``. Passing a whole
# batch at once held about seven times the batch per worker in decoded images
# and processor intermediates: 1 GiB for clip-vit-b32 at batch 256. Chunks of
# 32 cut that to about twice the batch, the chunks' outputs plus the assembled
# batch, at nearly the same speed; 16 and 64 were both slower.
PROCESSOR_CHUNK_SIZE = 32


class PreprocessMode(StrEnum):
    """Where the HuggingFace image processor runs inside the DataLoader workers."""

    # The processor runs in ``collate_fn`` on chunks of the batch; fast
    # (torchvision) processors vectorize each chunk and pad mixed sizes themselves.
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


def _encode_pool(size: ImageSize, pool_size: int, seed: int, quality: int) -> list[bytes]:
    rng = np.random.default_rng(seed)
    jpegs: list[bytes] = []
    for _ in range(pool_size):
        buffer = io.BytesIO()
        _smooth_random_image(rng, size).save(buffer, format="JPEG", quality=quality)
        jpegs.append(buffer.getvalue())
    return jpegs


def _find_images(root: Path, seed: int) -> list[Path]:
    files = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file())
    if not files:
        extensions = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise FileNotFoundError(f"no images ({extensions}) found under {root}")
    random.Random(seed).shuffle(files)  # sort first so the order depends only on the seed
    return files


# Image sources pickle as a recipe rather than their contents. Spawned
# DataLoader workers (the default on macOS and Windows) receive the dataset
# through a pipe. A payload larger than the pipe buffer (64 KiB) blocks the
# parent until that worker has imported torch, so workers would start one at
# a time: about 1.6 s each, on every trial. Forked workers (Linux) share memory
# and never unpickle, so they are unaffected.


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
        self.pool_size = pool_size
        self.seed = seed
        self.quality = quality
        self._jpegs = _encode_pool(size, pool_size, seed, quality)

    def __getstate__(self) -> dict[str, Any]:
        # The seed rebuilds a byte-identical pool, so ship only the recipe.
        return {"size": self.size, "pool_size": self.pool_size, "seed": self.seed, "quality": self.quality}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.size = state["size"]
        self.pool_size = state["pool_size"]
        self.seed = state["seed"]
        self.quality = state["quality"]
        self._jpegs = _encode_pool(self.size, self.pool_size, self.seed, self.quality)

    def __len__(self) -> int:
        return len(self._jpegs)

    def load(self, index: int) -> Image.Image:
        """Decode image ``index`` (modulo the pool size)."""
        with Image.open(io.BytesIO(self._jpegs[index % len(self._jpegs)])) as image:
            return image.convert("RGB")


class FolderImageSource:
    """Images found recursively under ``root``.

    Files are visited in a seeded random order. Sorted order would give a
    small batch size images from the first subdirectory only (one class of an
    ImageNet-style tree), which is not representative of sizes or content.

    Args:
        root: Directory to search.
        resize: Optional ``(width, height)`` every image is resized to, which
            makes per-sample preprocessing produce equal-sized tensors.
        seed: Seed for the visiting order.

    Raises:
        FileNotFoundError: If ``root`` holds no supported images.
    """

    def __init__(self, root: Path, resize: ImageSize | None = None, seed: int = 0) -> None:
        self.root = root
        self.resize = resize
        self.seed = seed
        self.files = _find_images(root, seed)

    def __getstate__(self) -> dict[str, Any]:
        # A large tree pickles to megabytes of paths; rescanning is cheaper.
        return {"root": self.root, "resize": self.resize, "seed": self.seed}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.root = state["root"]
        self.resize = state["resize"]
        self.seed = state["seed"]
        self.files = _find_images(self.root, self.seed)

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

    def __getitems__(self, indices: list[int]) -> LazyImages:
        """One batch's images, decoded when read rather than all up front.

        The DataLoader passes whatever this returns to ``collate_fn``, which
        :class:`BatchProcessorCollator` reads a chunk at a time. Decoding still
        happens in the worker and costs the same, but only one chunk of
        decoded pixels is held at once instead of the whole batch.
        """
        return LazyImages(self, indices)


class LazyImages(Sequence[Image.Image]):
    """The images at ``indices`` in ``dataset``, decoded on every access."""

    def __init__(self, dataset: ImageDataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    @overload
    def __getitem__(self, index: int) -> Image.Image: ...

    @overload
    def __getitem__(self, index: slice) -> list[Image.Image]: ...

    def __getitem__(self, index: int | slice) -> Image.Image | list[Image.Image]:
        if isinstance(index, slice):
            return [self.dataset[i] for i in self.indices[index]]
        return self.dataset[self.indices[index]]


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
    """``collate_fn`` that preprocesses a batch in chunks (``--preprocess batch``).

    The processor sees ``chunk_size`` images per call, so its intermediates
    scale with the chunk rather than the batch. Processors that keep aspect
    ratio (DETR, YOLOS) pad each call to its largest image, so chunks of mixed
    sizes come back in different shapes; they are zero-padded to the largest
    at the bottom and right, as the processor pads a single call.
    """

    def __init__(self, processor: ImageProcessor, chunk_size: int = PROCESSOR_CHUNK_SIZE) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        self.processor = processor
        self.chunk_size = chunk_size

    def __call__(self, images: Sequence[Image.Image]) -> dict[str, torch.Tensor]:
        """Preprocess ``images`` into one batch of tensors."""
        if not images:
            raise ValueError("cannot collate an empty batch")
        chunks = [
            _tensors_only(self.processor(images=list(images[start : start + self.chunk_size]), return_tensors="pt"))
            for start in range(0, len(images), self.chunk_size)
        ]
        return _assemble({key: [chunk[key] for chunk in chunks] for key in chunks[0]})


def pad_collate(samples: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack per-sample tensors, zero-padding trailing dimensions when shapes differ."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    return _assemble({key: [sample[key].unsqueeze(0) for sample in samples] for key in samples[0]})


def _assemble(parts: Mapping[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Concatenate each key's parts along dimension 0 into one batch tensor.

    Inside a DataLoader worker the values are :class:`_PaddedConcat`
    placeholders that become tensors on the way to the main process.
    """
    batch = {key: _PaddedConcat(key, tensors) for key, tensors in parts.items()}
    if get_worker_info() is None:
        return {key: concat.run() for key, concat in batch.items()}
    return cast("dict[str, torch.Tensor]", batch)


class _PaddedConcat:
    """Tensors to concatenate along dimension 0, zero-padding trailing dimensions to the largest.

    Inside a DataLoader worker this runs when the batch is pickled. Workers
    send batches through a multiprocessing queue whose feeder thread pickles
    them, and PyTorch copies each tensor into shared memory there. Building the
    batch straight into shared memory at that point replaces that copy, and
    leaves the worker free to start its next batch. Concatenating in
    ``collate_fn`` instead was 9% slower for clip-vit-b32 with 4 workers and
    20% slower for DETR with 2, where spare cores ran the feeder threads.
    Shapes are checked here, up front, because an error while pickling is only
    logged by the feeder thread and the batch never arrives.
    """

    def __init__(self, key: str, parts: list[torch.Tensor]) -> None:
        first = parts[0]
        if any(part.dim() != first.dim() for part in parts):
            raise ValueError(f"cannot pad {key!r}: samples have different ranks")
        trailing = (max(part.shape[dim] for part in parts) for dim in range(1, first.dim()))
        self.shape = (sum(len(part) for part in parts), *trailing)
        self.parts = parts

    def run(self) -> torch.Tensor:
        """Build the batch tensor, once: the parts are released so they free before the next key is built."""
        parts, self.parts = self.parts, []
        out = _new_batch(parts[0], self.shape)
        if all(part.shape[1:] == out.shape[1:] for part in parts):
            return torch.cat(parts, out=out)
        row = 0
        for part in parts:
            rows = out[row : row + len(part)]
            if part.shape[1:] != out.shape[1:]:
                rows.zero_()
            rows[(slice(None), *(slice(0, size) for size in part.shape[1:]))] = part
            row += len(part)
        return out

    def __reduce__(self) -> tuple[Callable[[torch.Tensor], torch.Tensor], tuple[torch.Tensor]]:
        return _identity, (self.run(),)


def _identity(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


def _new_batch(like: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """An uninitialized tensor for a batch, in shared memory inside a DataLoader worker.

    Workers hand batches to the main process through shared memory, so
    building the batch there, as ``default_collate`` does, saves a copy.
    """
    if get_worker_info() is None:
        return like.new_empty(shape)
    storage = like._typed_storage()._new_shared(math.prod(shape), device=like.device)
    return like.new(storage).resize_(shape)


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

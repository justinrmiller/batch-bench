from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from batch_bench.data import (
    BatchProcessorCollator,
    FolderImageSource,
    ImageDataset,
    PreprocessedDataset,
    PreprocessMode,
    SyntheticImageSource,
    build_loader,
    pad_collate,
    parse_image_size,
)


class FakeProcessor:
    """Resizes to 8x8 and returns a non-tensor field the collator must drop."""

    def __call__(self, images: Image.Image | list[Image.Image], return_tensors: str) -> Mapping[str, Any]:
        assert return_tensors == "pt"
        batch = images if isinstance(images, list) else [images]
        pixels = [torch.from_numpy(np.array(image.resize((8, 8)))).permute(2, 0, 1).float() for image in batch]
        return {"pixel_values": torch.stack(pixels), "sizes": [image.size for image in batch]}


@pytest.mark.parametrize(("text", "expected"), [("640x480", (640, 480)), ("32X16", (32, 16)), (" 8 x 4 ", (8, 4))])
def test_parse_image_size(text: str, expected: tuple[int, int]):
    assert parse_image_size(text) == expected


@pytest.mark.parametrize("text", ["640", "640x", "x480", "axb", "0x10", "-1x4", "1x2x3"])
def test_parse_image_size_rejects_bad_input(text: str):
    with pytest.raises(ValueError, match="image size"):
        parse_image_size(text)


class TestSyntheticImageSource:
    def test_images_have_the_requested_size_and_cycle(self):
        source = SyntheticImageSource(size=(40, 30), pool_size=3)
        assert len(source) == 3
        image = source.load(0)
        assert image.size == (40, 30)
        assert image.mode == "RGB"
        assert np.array_equal(np.asarray(source.load(4)), np.asarray(source.load(1)))

    def test_seed_makes_images_reproducible(self):
        first = np.asarray(SyntheticImageSource(size=(32, 32), pool_size=1, seed=7).load(0))
        same = np.asarray(SyntheticImageSource(size=(32, 32), pool_size=1, seed=7).load(0))
        other = np.asarray(SyntheticImageSource(size=(32, 32), pool_size=1, seed=8).load(0))
        assert np.array_equal(first, same)
        assert not np.array_equal(first, other)

    def test_pickles_as_a_small_recipe_that_rebuilds_the_same_pool(self):
        source = SyntheticImageSource(pool_size=32)
        payload = pickle.dumps(source)
        # Must fit a pipe buffer, or spawned DataLoader workers start one at a time.
        assert len(payload) < 4096
        clone = pickle.loads(payload)
        assert clone._jpegs == source._jpegs
        assert (clone.size, len(clone)) == (source.size, 32)

    def test_pool_must_not_be_empty(self):
        with pytest.raises(ValueError, match="pool_size"):
            SyntheticImageSource(pool_size=0)


class TestFolderImageSource:
    @pytest.fixture
    def image_dir(self, tmp_path: Path) -> Path:
        (tmp_path / "nested").mkdir()
        Image.new("RGB", (20, 10), "red").save(tmp_path / "a.png")
        Image.new("L", (12, 12), 128).save(tmp_path / "nested" / "b.JPG")
        (tmp_path / "notes.txt").write_text("not an image")
        return tmp_path

    def test_finds_images_recursively(self, image_dir: Path):
        source = FolderImageSource(image_dir)
        assert sorted(path.name for path in source.files) == ["a.png", "b.JPG"]
        assert len(source) == 2
        assert {source.load(i).size for i in range(2)} == {(20, 10), (12, 12)}
        assert source.load(2).size == source.load(0).size  # wraps around
        assert all(source.load(i).mode == "RGB" for i in range(2))

    def test_order_is_shuffled_by_seed_not_sorted(self, tmp_path: Path):
        # Sorted order would feed a small batch from one subdirectory only.
        for folder in ("class_a", "class_b", "class_c"):
            (tmp_path / folder).mkdir()
            for index in range(10):
                Image.new("RGB", (4, 4)).save(tmp_path / folder / f"{index}.png")
        first = FolderImageSource(tmp_path, seed=1)
        assert first.files == FolderImageSource(tmp_path, seed=1).files
        assert first.files != FolderImageSource(tmp_path, seed=2).files
        assert first.files != sorted(first.files)
        assert len({path.parent.name for path in first.files[:6]}) > 1
        assert pickle.loads(pickle.dumps(first)).files == first.files

    def test_resize(self, image_dir: Path):
        source = FolderImageSource(image_dir, resize=(16, 16))
        assert {source.load(i).size for i in range(2)} == {(16, 16)}

    def test_pickles_without_the_file_list(self, image_dir: Path):
        source = FolderImageSource(image_dir, resize=(16, 16))
        payload = pickle.dumps(source)
        assert b"b.JPG" not in payload
        clone = pickle.loads(payload)
        assert clone.files == source.files
        assert clone.resize == (16, 16)

    def test_empty_directory_is_an_error(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="no images"):
            FolderImageSource(tmp_path)


def test_image_dataset_cycles_and_bounds_checks(tiny_source: SyntheticImageSource):
    dataset = ImageDataset(tiny_source, length=10)
    assert len(dataset) == 10
    assert np.array_equal(np.asarray(dataset[5]), np.asarray(tiny_source.load(1)))
    with pytest.raises(IndexError):
        dataset[10]


def test_preprocessed_dataset_drops_the_batch_dimension(tiny_source: SyntheticImageSource):
    dataset = PreprocessedDataset(ImageDataset(tiny_source, 3), FakeProcessor())
    assert len(dataset) == 3
    sample = dataset[0]
    assert set(sample) == {"pixel_values"}
    assert sample["pixel_values"].shape == (3, 8, 8)


def test_batch_collator_keeps_only_tensors(tiny_source: SyntheticImageSource):
    batch = BatchProcessorCollator(FakeProcessor())([tiny_source.load(0), tiny_source.load(1)])
    assert set(batch) == {"pixel_values"}
    assert batch["pixel_values"].shape == (2, 3, 8, 8)


class TestPadCollate:
    def test_equal_shapes_are_stacked(self):
        batch = pad_collate([{"x": torch.ones(2, 3)}, {"x": torch.zeros(2, 3)}])
        assert batch["x"].shape == (2, 2, 3)

    def test_mixed_shapes_are_zero_padded(self):
        batch = pad_collate([{"x": torch.ones(3, 2, 4)}, {"x": torch.full((3, 4, 2), 2.0)}])
        padded = batch["x"]
        assert padded.shape == (2, 3, 4, 4)
        assert torch.equal(padded[0, :, :2, :4], torch.ones(3, 2, 4))
        assert padded[0, :, 2:, :].abs().sum() == 0
        assert torch.equal(padded[1, :, :4, :2], torch.full((3, 4, 2), 2.0))
        assert padded[1, :, :, 2:].abs().sum() == 0

    def test_mixed_ranks_are_an_error(self):
        with pytest.raises(ValueError, match="different ranks"):
            pad_collate([{"x": torch.ones(2)}, {"x": torch.ones(2, 2)}])

    def test_empty_batch_is_an_error(self):
        with pytest.raises(ValueError, match="empty"):
            pad_collate([])


@pytest.mark.parametrize("mode", list(PreprocessMode))
def test_build_loader_yields_full_batches(tiny_source: SyntheticImageSource, mode: PreprocessMode):
    loader = build_loader(ImageDataset(tiny_source, 7), FakeProcessor(), batch_size=3, num_workers=0, preprocess=mode)
    batches = list(loader)
    assert len(batches) == 2  # drop_last discards the partial seventh image
    assert all(batch["pixel_values"].shape == (3, 3, 8, 8) for batch in batches)
    assert loader.prefetch_factor is None


def test_build_loader_with_workers(tiny_source: SyntheticImageSource):
    loader = build_loader(ImageDataset(tiny_source, 4), FakeProcessor(), batch_size=2, num_workers=1, prefetch_factor=3)
    assert loader.prefetch_factor == 3
    assert [batch["pixel_values"].shape[0] for batch in loader] == [2, 2]

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import transformers
from transformers import CLIPConfig, CLIPModel

from batch_bench import models
from batch_bench.models import (
    DEFAULT_PRESETS,
    PRESETS,
    ModelSpec,
    Task,
    infer_task,
    load_model,
    make_forward,
    primary_output,
    resolve_model,
    surrogate_loss,
)


class TestPresets:
    def test_every_task_has_a_default_preset_of_that_task(self):
        assert set(DEFAULT_PRESETS) == set(Task)
        for task, name in DEFAULT_PRESETS.items():
            assert PRESETS[name].task is task

    def test_every_task_has_several_presets(self):
        for task in Task:
            assert sum(spec.task is task for spec in PRESETS.values()) >= 4

    def test_preset_keys_and_classes_are_valid(self):
        for name, spec in PRESETS.items():
            assert spec.name == name
            assert spec.description
            assert spec.model_class is not None
            assert hasattr(transformers, spec.model_class), spec.model_class


@pytest.mark.parametrize(
    ("architectures", "expected"),
    [
        (["DetrForObjectDetection"], Task.OBJECT_DETECTION),
        (["SegformerForSemanticSegmentation"], Task.IMAGE_SEGMENTATION),
        (["Mask2FormerForUniversalSegmentation"], Task.IMAGE_SEGMENTATION),
        (["CLIPModel"], Task.EMBEDDING),
        (["CLIPVisionModelWithProjection"], Task.EMBEDDING),
        (["DINOv3ViTModel"], Task.EMBEDDING),
        (["ViTForImageClassification"], None),
        ([], None),
    ],
)
def test_infer_task(architectures: list[str], expected: Task | None):
    assert infer_task(architectures) is expected


class TestResolveModel:
    def test_preset_names_resolve_without_network(self):
        assert resolve_model("segformer-b0") is PRESETS["segformer-b0"]

    def test_task_is_inferred_from_a_local_config(self, tiny_detector_dir: Path):
        spec = resolve_model(str(tiny_detector_dir))
        assert spec.task is Task.OBJECT_DETECTION
        assert spec.model_id == str(tiny_detector_dir)
        assert spec.name == tiny_detector_dir.name
        assert spec.model_class == "YolosForObjectDetection"

    def test_checkpoint_architecture_keeps_projection_heads(self, tiny_embedder_dir: Path):
        loaded = load_model(resolve_model(str(tiny_embedder_dir)))
        assert isinstance(loaded.model, transformers.CLIPVisionModelWithProjection)

    def test_explicit_task_wins(self, tiny_embedder_dir: Path):
        spec = resolve_model(str(tiny_embedder_dir), task=Task.IMAGE_SEGMENTATION)
        assert spec.task is Task.IMAGE_SEGMENTATION
        # The checkpoint's own class does a different task, so the task's auto classes are used.
        assert spec.model_class is None

    def test_embedding_task_on_a_classifier_uses_the_backbone(self, tmp_path: Path):
        config = transformers.ViTConfig(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            image_size=32,
            patch_size=8,
        )
        transformers.ViTForImageClassification(config).save_pretrained(tmp_path)
        transformers.ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(tmp_path)
        spec = resolve_model(str(tmp_path), task=Task.EMBEDDING)
        assert spec.model_class is None
        loaded = load_model(spec)
        assert isinstance(loaded.model, transformers.ViTModel)
        output = make_forward(Task.EMBEDDING, loaded.model.eval())(_pixels(2, size=32))
        assert primary_output(Task.EMBEDDING, output).shape == (2, 16)

    def test_uninferable_task_is_an_error(self, tmp_path: Path):
        config = transformers.ViTConfig(hidden_size=8, num_hidden_layers=1, num_attention_heads=1)
        config.architectures = ["ViTForImageClassification"]
        config.save_pretrained(tmp_path)
        assert json.loads((tmp_path / "config.json").read_text())["architectures"]
        with pytest.raises(ValueError, match="pass --task"):
            resolve_model(str(tmp_path))


class TestLoadModel:
    def test_loads_float32_weights_and_the_processor(self, tiny_segmenter_dir: Path):
        spec = ModelSpec("seg", str(tiny_segmenter_dir), Task.IMAGE_SEGMENTATION, "AutoModelForSemanticSegmentation")
        loaded = load_model(spec)
        assert isinstance(loaded.model, transformers.SegformerForSemanticSegmentation)
        assert next(loaded.model.parameters()).dtype == torch.float32
        assert loaded.num_parameters == sum(p.numel() for p in loaded.model.parameters())
        assert callable(loaded.processor)

    def test_segmentation_falls_back_to_universal_models(self, tiny_universal_segmenter_dir: Path):
        spec = ModelSpec("m2f", str(tiny_universal_segmenter_dir), Task.IMAGE_SEGMENTATION)
        loaded = load_model(spec)
        assert isinstance(loaded.model, transformers.Mask2FormerForUniversalSegmentation)

    def test_wrong_task_reports_every_attempt(self, tiny_segmenter_dir: Path):
        spec = ModelSpec("seg", str(tiny_segmenter_dir), Task.OBJECT_DETECTION)
        with pytest.raises(ValueError, match="AutoModelForObjectDetection"):
            load_model(spec)


def _pixels(batch: int, size: int = 64) -> dict[str, torch.Tensor]:
    return {"pixel_values": torch.randn(batch, 3, size, size)}


class TestForward:
    def test_unknown_inputs_are_dropped(self, tiny_detector_dir: Path):
        loaded = load_model(resolve_model(str(tiny_detector_dir)))
        forward = make_forward(Task.OBJECT_DETECTION, loaded.model.eval())
        # YOLOS takes no pixel_mask; passing it through would raise a TypeError.
        output = forward({**_pixels(2), "pixel_mask": torch.ones(2, 64, 64, dtype=torch.long)})
        assert primary_output(Task.OBJECT_DETECTION, output).shape[0] == 2

    def test_full_clip_runs_only_the_image_tower(self):
        config = CLIPConfig(
            text_config={
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "vocab_size": 50000,
            },
            vision_config={
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "image_size": 32,
                "patch_size": 8,
            },
            projection_dim=8,
        )
        model = CLIPModel(config).eval()
        output = make_forward(Task.EMBEDDING, model)(_pixels(3, size=32))
        assert primary_output(Task.EMBEDDING, output).shape == (3, 8)

    def test_compile_wraps_the_target(self, tiny_embedder_dir: Path, monkeypatch: pytest.MonkeyPatch):
        compiled: list[object] = []

        def fake_compile(target: object) -> object:
            compiled.append(target)
            return target

        monkeypatch.setattr(torch, "compile", fake_compile)
        loaded = load_model(resolve_model(str(tiny_embedder_dir)))
        forward = make_forward(Task.EMBEDDING, loaded.model.eval(), compile_model=True)
        assert compiled == [loaded.model]
        assert primary_output(Task.EMBEDDING, forward(_pixels(2, size=32))).shape == (2, 16)


class TestSurrogateLoss:
    def test_sums_the_means_of_nested_float_tensors(self):
        output = {
            "logits": torch.full((2, 3), 2.0),
            "aux": [{"boxes": torch.full((2,), 1.0)}, (torch.full((4,), 3.0),)],
            "ids": torch.tensor([5, 7]),  # integer tensors are ignored
            "note": "not a tensor",
        }
        assert surrogate_loss(output).item() == pytest.approx(6.0)

    def test_backpropagates_to_every_output(self):
        a = torch.ones(2, requires_grad=True)
        b = torch.ones(3, requires_grad=True)
        surrogate_loss({"a": a * 2, "b": [b * 3]}).backward()
        assert a.grad is not None
        assert b.grad is not None

    def test_requires_a_float_tensor(self):
        with pytest.raises(ValueError, match="no floating-point"):
            surrogate_loss({"ids": torch.tensor([1])})


class TestPrimaryOutput:
    def test_tensors_pass_through(self):
        tensor = torch.zeros(2, 4)
        assert primary_output(Task.EMBEDDING, tensor) is tensor

    def test_embedding_prefers_projected_embeds(self):
        embeds = torch.zeros(2, 8)
        output = {"image_embeds": embeds, "last_hidden_state": torch.zeros(2, 5, 4)}
        assert primary_output(Task.EMBEDDING, output) is embeds

    def test_embedding_without_pooler_averages_tokens(self):
        hidden = torch.arange(12.0).reshape(1, 3, 4)
        result = primary_output(Task.EMBEDDING, SimpleNamespace(last_hidden_state=hidden))
        assert torch.equal(result, hidden.mean(dim=1))

    def test_universal_segmentation_uses_mask_logits(self):
        masks = torch.zeros(2, 4, 16, 16)
        output = {"class_queries_logits": torch.zeros(2, 4, 3), "masks_queries_logits": masks}
        assert primary_output(Task.IMAGE_SEGMENTATION, output) is masks

    def test_missing_fields_are_an_error(self):
        with pytest.raises(ValueError, match="object-detection"):
            primary_output(Task.OBJECT_DETECTION, {"something": torch.zeros(1)})


def test_accepted_inputs_ignores_var_kwargs():
    def target(pixel_values: torch.Tensor, *, pixel_mask: torch.Tensor | None = None, **kwargs: object) -> None:
        del pixel_values, pixel_mask, kwargs

    assert models._accepted_inputs(target) == frozenset({"pixel_values", "pixel_mask"})

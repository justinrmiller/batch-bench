"""HuggingFace model presets, loading, and per-task forward passes."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
import transformers
from transformers import AutoConfig, AutoImageProcessor

from batch_bench.data import ImageProcessor


class Task(StrEnum):
    """Vision workloads the benchmark covers."""

    OBJECT_DETECTION = "object-detection"
    IMAGE_SEGMENTATION = "image-segmentation"
    EMBEDDING = "embedding"


@dataclass(frozen=True)
class ModelSpec:
    """A checkpoint to benchmark.

    Attributes:
        name: Short name used in reports.
        model_id: HuggingFace Hub id or local directory.
        task: Workload the model performs.
        model_class: ``transformers`` class used to load it; None tries the
            task's auto classes in order.
        description: One-line summary for ``batch-bench models``.
    """

    name: str
    model_id: str
    task: Task
    model_class: str | None = None
    description: str = ""


_OD = Task.OBJECT_DETECTION
_SEG = Task.IMAGE_SEGMENTATION
_EMB = Task.EMBEDDING

PRESETS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec(
            "detr-resnet-50",
            "facebook/detr-resnet-50",
            _OD,
            "AutoModelForObjectDetection",
            "DETR, ResNet-50 backbone, COCO; 800px shortest edge",
        ),
        ModelSpec(
            "rtdetr-v2-r18",
            "PekingU/rtdetr_v2_r18vd",
            _OD,
            "AutoModelForObjectDetection",
            "RT-DETRv2 real-time detector, ResNet-18, 640x640",
        ),
        ModelSpec(
            "rtdetr-r50",
            "PekingU/rtdetr_r50vd",
            _OD,
            "AutoModelForObjectDetection",
            "RT-DETR real-time detector, ResNet-50, 640x640",
        ),
        ModelSpec(
            "dfine-small",
            "ustc-community/dfine-small-coco",
            _OD,
            "AutoModelForObjectDetection",
            "D-FINE small real-time detector, COCO, 640x640",
        ),
        ModelSpec(
            "yolos-small",
            "hustvl/yolos-small",
            _OD,
            "AutoModelForObjectDetection",
            "YOLOS ViT-S detector, COCO",
        ),
        ModelSpec(
            "segformer-b0",
            "nvidia/segformer-b0-finetuned-ade-512-512",
            _SEG,
            "AutoModelForSemanticSegmentation",
            "SegFormer-B0 semantic segmentation, ADE20k, 512x512",
        ),
        ModelSpec(
            "segformer-b5",
            "nvidia/segformer-b5-finetuned-ade-640-640",
            _SEG,
            "AutoModelForSemanticSegmentation",
            "SegFormer-B5 semantic segmentation, ADE20k, 640x640",
        ),
        ModelSpec(
            "upernet-convnext-small",
            "openmmlab/upernet-convnext-small",
            _SEG,
            "AutoModelForSemanticSegmentation",
            "UperNet + ConvNeXt-S semantic segmentation, ADE20k",
        ),
        ModelSpec(
            "mask2former-swin-tiny",
            "facebook/mask2former-swin-tiny-coco-instance",
            _SEG,
            "AutoModelForUniversalSegmentation",
            "Mask2Former + Swin-T instance segmentation, COCO",
        ),
        ModelSpec(
            "clip-vit-b32",
            "openai/clip-vit-base-patch32",
            _EMB,
            "CLIPVisionModelWithProjection",
            "CLIP ViT-B/32 image embeddings (vision tower only), 224x224",
        ),
        ModelSpec(
            "clip-vit-l14",
            "openai/clip-vit-large-patch14",
            _EMB,
            "CLIPVisionModelWithProjection",
            "CLIP ViT-L/14 image embeddings (vision tower only), 224x224",
        ),
        ModelSpec(
            "siglip2-base",
            "google/siglip2-base-patch16-224",
            _EMB,
            "SiglipVisionModel",
            "SigLIP 2 ViT-B/16 image embeddings (vision tower only), 224x224",
        ),
        ModelSpec(
            "dinov2-small",
            "facebook/dinov2-small",
            _EMB,
            "AutoModel",
            "DINOv2 ViT-S/14 self-supervised image features",
        ),
        ModelSpec(
            "dinov2-base",
            "facebook/dinov2-base",
            _EMB,
            "AutoModel",
            "DINOv2 ViT-B/14 self-supervised image features",
        ),
    )
}

# Benchmarked when the user names a task but no model.
DEFAULT_PRESETS: dict[Task, str] = {
    Task.OBJECT_DETECTION: "detr-resnet-50",
    Task.IMAGE_SEGMENTATION: "segformer-b0",
    Task.EMBEDDING: "clip-vit-b32",
}

# Auto classes tried, in order, for checkpoints that are not presets.
_TASK_AUTO_CLASSES: dict[Task, tuple[str, ...]] = {
    Task.OBJECT_DETECTION: ("AutoModelForObjectDetection",),
    Task.IMAGE_SEGMENTATION: (
        "AutoModelForSemanticSegmentation",
        "AutoModelForUniversalSegmentation",
        "AutoModelForImageSegmentation",
    ),
    Task.EMBEDDING: ("AutoModel",),
}


def infer_task(architectures: Sequence[str]) -> Task | None:
    """Guess the task from a config's ``architectures`` list."""
    for arch in architectures:
        if "ObjectDetection" in arch:
            return Task.OBJECT_DETECTION
        if "Segmentation" in arch:
            return Task.IMAGE_SEGMENTATION
    # Bare backbones (CLIPModel, Dinov2Model, SiglipVisionModel, ...) produce embeddings.
    if any(arch.endswith(("Model", "WithProjection")) for arch in architectures):
        return Task.EMBEDDING
    return None


def resolve_model(name: str, task: Task | None = None, revision: str | None = None) -> ModelSpec:
    """Turn a preset name or Hub id into a :class:`ModelSpec`.

    Non-preset checkpoints load with the class named in their config's
    ``architectures`` when transformers has it. Auto classes can pick a
    different head: ``AutoModel`` turns a ``CLIPVisionModelWithProjection``
    checkpoint into a ``CLIPVisionModel`` and silently drops the projection.

    Args:
        name: Preset name (see :data:`PRESETS`), Hub id, or local directory.
        task: Task for non-preset checkpoints; inferred from the config when None.
        revision: Hub revision of the config.

    Raises:
        ValueError: If the task cannot be inferred.
    """
    if name in PRESETS:
        return PRESETS[name]
    config = AutoConfig.from_pretrained(name, revision=revision)
    architectures: list[str] = list(getattr(config, "architectures", None) or [])
    task = task or infer_task(architectures)
    if task is None:
        raise ValueError(f"cannot infer the task for {name!r} from its config; pass --task ({', '.join(Task)})")
    model_class = next((arch for arch in architectures if hasattr(transformers, arch)), None)
    short = name.rstrip("/").rsplit("/", 1)[-1]
    return ModelSpec(name=short, model_id=name, task=task, model_class=model_class)


@dataclass
class LoadedModel:
    """A model in float32 on the CPU, plus its image processor."""

    spec: ModelSpec
    model: torch.nn.Module
    processor: ImageProcessor

    @property
    def num_parameters(self) -> int:
        """Total parameter count."""
        return sum(p.numel() for p in self.model.parameters())


def load_model(spec: ModelSpec, revision: str | None = None) -> LoadedModel:
    """Load ``spec``'s weights (float32, CPU) and image processor.

    Weights stay in float32: mixed precision comes from autocast, which keeps
    training numerically sane and works on every backend.
    """
    class_names = (spec.model_class,) if spec.model_class else _TASK_AUTO_CLASSES[spec.task]
    errors: list[str] = []
    model: torch.nn.Module | None = None
    for class_name in class_names:
        model_cls = getattr(transformers, class_name)
        try:
            model = model_cls.from_pretrained(spec.model_id, revision=revision, dtype=torch.float32)
            break
        except ValueError as exc:  # transformers' "unrecognized configuration class"
            errors.append(f"{class_name}: {exc}")
    if model is None:
        raise ValueError(f"no {spec.task} model class can load {spec.model_id!r}: " + "; ".join(errors))
    processor = AutoImageProcessor.from_pretrained(spec.model_id, revision=revision)
    return LoadedModel(spec=spec, model=model, processor=processor)


ForwardFn = Callable[[Mapping[str, torch.Tensor]], object]


def _accepted_inputs(fn: Callable[..., Any]) -> frozenset[str]:
    """Named parameters of ``fn``; ``**kwargs`` is ignored so stray keys never leak in."""
    params = inspect.signature(fn).parameters.values()
    named = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return frozenset(p.name for p in params if p.kind in named)


def make_forward(task: Task, model: torch.nn.Module, *, compile_model: bool = False) -> ForwardFn:
    """Build the callable that runs one preprocessed batch through ``model``.

    Embedding models with a text tower (e.g. a full ``CLIPModel``) run only the
    image path via ``get_image_features``. Batch keys the target does not take
    by name (such as ``pixel_mask`` for YOLOS) are dropped.
    """
    get_features = getattr(model, "get_image_features", None)
    target: Callable[..., Any] = get_features if task is Task.EMBEDDING and callable(get_features) else model
    accepted = _accepted_inputs(target if target is not model else model.forward)
    if compile_model:
        target = torch.compile(target)

    def forward(batch: Mapping[str, torch.Tensor]) -> object:
        return target(**{key: value for key, value in batch.items() if key in accepted})

    return forward


def _floating_tensors(output: object) -> Iterator[torch.Tensor]:
    """Yield every floating-point tensor in a (possibly nested) model output."""
    if isinstance(output, torch.Tensor):
        if output.is_floating_point():
            yield output
    elif isinstance(output, Mapping):  # ModelOutput is a Mapping
        for value in output.values():
            yield from _floating_tensors(value)
    elif isinstance(output, list | tuple):
        for value in output:
            yield from _floating_tensors(value)


def surrogate_loss(output: object) -> torch.Tensor:
    """A cheap scalar that depends on every floating-point output.

    Backpropagating it exercises every head, so training-mode memory and
    compute match a real loss except for task-specific matching (e.g. DETR's
    Hungarian matcher), which runs mostly on the CPU anyway.
    """
    tensors = list(_floating_tensors(output))
    if not tensors:
        raise ValueError("model output contains no floating-point tensors")
    return torch.stack([tensor.float().mean() for tensor in tensors]).sum()


def primary_output(task: Task, output: object) -> torch.Tensor:
    """The tensor a user of ``task`` cares about: logits, masks, or embeddings."""
    if isinstance(output, torch.Tensor):
        return output
    if task is Task.EMBEDDING:
        keys: tuple[str, ...] = ("image_embeds", "pooler_output", "last_hidden_state")
    elif task is Task.IMAGE_SEGMENTATION:
        keys = ("logits", "masks_queries_logits", "pred_masks")
    else:
        keys = ("logits", "pred_boxes")
    for key in keys:
        value = output.get(key) if isinstance(output, Mapping) else getattr(output, key, None)
        if isinstance(value, torch.Tensor):
            # ViT-style backbones without a pooler: average the patch tokens.
            if task is Task.EMBEDDING and key == "last_hidden_state":
                return value.mean(dim=1)
            return value
    raise ValueError(f"cannot find a {task} output among {type(output).__name__} fields")

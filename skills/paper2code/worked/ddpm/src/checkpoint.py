"""Bounded, non-pickle DDPM checkpoint persistence.

Tensor data is stored in safetensors. Reproducibility/configuration data lives in
a separate, strictly validated JSON file with the same stem.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import stat
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from model import UNet, UNetConfig


FORMAT_NAME = "paper2code-ddpm-safetensors"
FORMAT_VERSION = 1
MAX_CHECKPOINT_BYTES = 1_100_000_000
MAX_METADATA_BYTES = 256_000
MAX_HEADER_BYTES = 1_000_000
MAX_TENSORS = 4_096
MAX_TENSOR_RANK = 8
MAX_TENSOR_ELEMENTS = 100_000_000
MAX_TOTAL_ELEMENTS = 275_000_000
MAX_TENSOR_BYTES = 400_000_000
MAX_TOTAL_TENSOR_BYTES = 1_100_000_000
MAX_KEY_CHARS = 512

_DTYPE_BYTES = {"F32": 4}
_TORCH_DTYPE_TO_SAFE = {torch.float32: "F32"}


class CheckpointSecurityError(ValueError):
    """The checkpoint violated its serialization or resource boundary."""


def _is_link_or_reparse_point(path: Path) -> bool:
    metadata = path.lstat()
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _reject_linked_components(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if os.path.lexists(current) and _is_link_or_reparse_point(current):
            raise CheckpointSecurityError(
                "checkpoint paths must not contain links or reparse points"
            )


def _open_bounded_regular(path: Path, maximum: int) -> tuple[int, os.stat_result]:
    path = Path(os.path.abspath(path))
    _reject_linked_components(path)
    before = path.lstat()
    if _is_link_or_reparse_point(path) or not stat.S_ISREG(before.st_mode):
        raise CheckpointSecurityError("checkpoint component must be a regular non-link file")
    if before.st_size > maximum:
        raise CheckpointSecurityError("checkpoint component exceeds its file-size limit")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode) or opened.st_size > maximum:
        os.close(descriptor)
        raise CheckpointSecurityError("opened checkpoint component violates its file limit")
    if (
        getattr(before, "st_ino", 0)
        and getattr(opened, "st_ino", 0)
        and (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        os.close(descriptor)
        raise CheckpointSecurityError("checkpoint component changed while opening")
    return descriptor, opened


def _read_bounded_regular(path: Path, maximum: int) -> bytes:
    descriptor, _ = _open_bounded_regular(path, maximum)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            value = handle.read(maximum + 1)
    finally:
        os.close(descriptor)
    if len(value) > maximum:
        raise CheckpointSecurityError("checkpoint component exceeds its read limit")
    return value


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise CheckpointSecurityError(f"duplicate JSON field: {key}")
        output[key] = value
    return output


def _expect_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise CheckpointSecurityError(f"{label} fields differ from the strict schema")


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise CheckpointSecurityError(f"{label} is outside its integer range")
    return value


def _number(value: Any, label: str, minimum: float, maximum: float) -> float:
    if type(value) not in {int, float}:
        raise CheckpointSecurityError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise CheckpointSecurityError(f"{label} is outside its numeric range")
    return result


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise CheckpointSecurityError(f"{label} must be boolean")
    return value


def _short_text(value: Any, label: str, allowed: set[str] | None = None) -> str:
    if type(value) is not str or not value or len(value) > 256:
        raise CheckpointSecurityError(f"{label} must be bounded text")
    if allowed is not None and value not in allowed:
        raise CheckpointSecurityError(f"{label} is outside the supported values")
    return value


def validate_config(value: Any) -> dict[str, Any]:
    """Return a canonical DDPM config after strict schema/range validation."""
    if not isinstance(value, dict):
        raise CheckpointSecurityError("config must be an object")
    _expect_keys(value, {"diffusion", "model", "training", "data", "eval"}, "config")
    if not all(isinstance(value[name], dict) for name in value):
        raise CheckpointSecurityError("every config section must be an object")

    diffusion = value["diffusion"]
    _expect_keys(
        diffusion,
        {"timesteps", "beta_start", "beta_end", "schedule", "variance_type"},
        "diffusion config",
    )
    timesteps = _integer(diffusion["timesteps"], "diffusion.timesteps", 1, 10_000)
    beta_start = _number(diffusion["beta_start"], "diffusion.beta_start", 1e-8, 0.5)
    beta_end = _number(diffusion["beta_end"], "diffusion.beta_end", beta_start, 0.999)

    model = value["model"]
    _expect_keys(
        model,
        {
            "image_channels", "base_channels", "channel_mults", "num_res_blocks",
            "attention_resolutions", "dropout", "time_embed_dim", "num_groups",
            "image_size",
        },
        "model config",
    )
    image_channels = _integer(model["image_channels"], "model.image_channels", 1, 4)
    base_channels = _integer(model["base_channels"], "model.base_channels", 8, 256)
    image_size = _integer(model["image_size"], "model.image_size", 8, 256)
    if image_size & (image_size - 1):
        raise CheckpointSecurityError("model.image_size must be a power of two")
    channel_mults = model["channel_mults"]
    if (
        type(channel_mults) is not list
        or not 1 <= len(channel_mults) <= 6
        or any(type(item) is not int or not 1 <= item <= 8 for item in channel_mults)
    ):
        raise CheckpointSecurityError("model.channel_mults violates its collection budget")
    if image_size // (2 ** (len(channel_mults) - 1)) < 1:
        raise CheckpointSecurityError("model.channel_mults downsample past the image size")
    attention = model["attention_resolutions"]
    if (
        type(attention) is not list
        or len(attention) > 6
        or len(set(attention)) != len(attention)
        or any(type(item) is not int or not 1 <= item <= image_size for item in attention)
    ):
        raise CheckpointSecurityError("model.attention_resolutions violates its collection budget")
    num_groups = _integer(model["num_groups"], "model.num_groups", 1, 64)
    if any((base_channels * multiplier) % num_groups for multiplier in channel_mults):
        raise CheckpointSecurityError("model channels must be divisible by num_groups")

    training = value["training"]
    _expect_keys(
        training,
        {
            "optimizer", "lr", "betas", "eps", "weight_decay", "total_steps",
            "batch_size", "gradient_clip", "schedule", "ema_decay",
            "ema_start_step",
        },
        "training config",
    )
    betas = training["betas"]
    if type(betas) is not list or len(betas) != 2:
        raise CheckpointSecurityError("training.betas must contain exactly two numbers")
    normalized_betas = [
        _number(item, f"training.betas[{index}]", 0.0, 0.999999)
        for index, item in enumerate(betas)
    ]
    if normalized_betas[0] >= normalized_betas[1]:
        raise CheckpointSecurityError("training.betas must be strictly increasing")
    gradient_clip = training["gradient_clip"]
    if gradient_clip is not None:
        gradient_clip = _number(gradient_clip, "training.gradient_clip", 1e-12, 1e6)

    data = value["data"]
    _expect_keys(data, {"dataset", "image_size", "augmentation", "normalize"}, "data config")
    data_image_size = _integer(data["image_size"], "data.image_size", 8, 256)
    if data_image_size != image_size:
        raise CheckpointSecurityError("model and data image sizes must match")
    evaluation = value["eval"]
    _expect_keys(evaluation, {"metric", "num_samples", "sampling_steps"}, "eval config")
    sampling_steps = _integer(evaluation["sampling_steps"], "eval.sampling_steps", 1, timesteps)

    return {
        "diffusion": {
            "timesteps": timesteps,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "schedule": _short_text(diffusion["schedule"], "diffusion.schedule", {"linear"}),
            "variance_type": _short_text(
                diffusion["variance_type"], "diffusion.variance_type",
                {"fixed_small", "fixed_large"},
            ),
        },
        "model": {
            "image_channels": image_channels,
            "base_channels": base_channels,
            "channel_mults": list(channel_mults),
            "num_res_blocks": _integer(model["num_res_blocks"], "model.num_res_blocks", 1, 4),
            "attention_resolutions": list(attention),
            "dropout": _number(model["dropout"], "model.dropout", 0.0, 0.5),
            "time_embed_dim": _integer(model["time_embed_dim"], "model.time_embed_dim", 8, 2_048),
            "num_groups": num_groups,
            "image_size": image_size,
        },
        "training": {
            "optimizer": _short_text(training["optimizer"], "training.optimizer", {"adam"}),
            "lr": _number(training["lr"], "training.lr", 1e-12, 1.0),
            "betas": normalized_betas,
            "eps": _number(training["eps"], "training.eps", 1e-16, 1.0),
            "weight_decay": _number(training["weight_decay"], "training.weight_decay", 0.0, 100.0),
            "total_steps": _integer(training["total_steps"], "training.total_steps", 1, 100_000_000),
            "batch_size": _integer(training["batch_size"], "training.batch_size", 1, 1_000_000),
            "gradient_clip": gradient_clip,
            "schedule": _short_text(training["schedule"], "training.schedule", {"constant"}),
            "ema_decay": _number(training["ema_decay"], "training.ema_decay", 0.0, 0.999999999),
            "ema_start_step": _integer(
                training["ema_start_step"], "training.ema_start_step", 0, 100_000_000
            ),
        },
        "data": {
            "dataset": _short_text(data["dataset"], "data.dataset"),
            "image_size": data_image_size,
            "augmentation": _short_text(
                data["augmentation"], "data.augmentation", {"none", "random_flip"}
            ),
            "normalize": _boolean(data["normalize"], "data.normalize"),
        },
        "eval": {
            "metric": _short_text(evaluation["metric"], "eval.metric", {"FID"}),
            "num_samples": _integer(evaluation["num_samples"], "eval.num_samples", 1, 1_000_000),
            "sampling_steps": sampling_steps,
        },
    }


def _unet_config(config: Mapping[str, Any]) -> UNetConfig:
    model = config["model"]
    return UNetConfig(
        image_channels=model["image_channels"],
        base_channels=model["base_channels"],
        channel_mults=tuple(model["channel_mults"]),
        num_res_blocks=model["num_res_blocks"],
        attention_resolutions=tuple(model["attention_resolutions"]),
        dropout=model["dropout"],
        time_embed_dim=model["time_embed_dim"],
        num_groups=model["num_groups"],
        image_size=model["image_size"],
    )


def _parse_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            _read_bounded_regular(path, MAX_METADATA_BYTES),
            object_pairs_hook=_strict_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointSecurityError("checkpoint metadata is not valid strict JSON") from exc
    if not isinstance(value, dict):
        raise CheckpointSecurityError("checkpoint metadata must be an object")
    _expect_keys(
        value,
        {"format", "format_version", "step", "loss", "has_ema", "config"},
        "checkpoint metadata",
    )
    if value["format"] != FORMAT_NAME or value["format_version"] != FORMAT_VERSION:
        raise CheckpointSecurityError("unsupported checkpoint format")
    step = _integer(value["step"], "checkpoint step", 0, 100_000_000)
    has_ema = _boolean(value["has_ema"], "checkpoint has_ema")
    loss = value["loss"]
    if loss is not None:
        loss = _number(loss, "checkpoint loss", -1e12, 1e12)
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "step": step,
        "loss": loss,
        "has_ema": has_ema,
        "config": validate_config(value["config"]),
    }


def _inspect_header(path: Path) -> dict[str, dict[str, Any]]:
    descriptor, opened = _open_bounded_regular(path, MAX_CHECKPOINT_BYTES)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise CheckpointSecurityError("safetensors header prefix is truncated")
            header_size = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_size <= MAX_HEADER_BYTES:
                raise CheckpointSecurityError("safetensors header exceeds its size budget")
            if 8 + header_size > opened.st_size:
                raise CheckpointSecurityError("safetensors header exceeds the file")
            header_bytes = handle.read(header_size)
            if len(header_bytes) != header_size:
                raise CheckpointSecurityError("safetensors header is truncated")
    finally:
        os.close(descriptor)

    try:
        raw_header = json.loads(header_bytes, object_pairs_hook=_strict_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointSecurityError("safetensors header is invalid JSON") from exc
    if not isinstance(raw_header, dict) or "__metadata__" in raw_header:
        raise CheckpointSecurityError("safetensors header violates the strict schema")
    if not 1 <= len(raw_header) <= MAX_TENSORS:
        raise CheckpointSecurityError("safetensors tensor-count budget exceeded")

    body_size = opened.st_size - 8 - header_size
    ranges: list[tuple[int, int, str]] = []
    total_elements = 0
    total_bytes = 0
    normalized: dict[str, dict[str, Any]] = {}
    for key, item in raw_header.items():
        if (
            type(key) is not str
            or not key
            or len(key) > MAX_KEY_CHARS
            or not isinstance(item, dict)
        ):
            raise CheckpointSecurityError("safetensors tensor key violates its budget")
        _expect_keys(item, {"dtype", "shape", "data_offsets"}, "safetensors tensor")
        dtype = item["dtype"]
        if dtype not in _DTYPE_BYTES:
            raise CheckpointSecurityError("safetensors tensor dtype is unsupported")
        shape = item["shape"]
        if (
            type(shape) is not list
            or len(shape) > MAX_TENSOR_RANK
            or any(type(size) is not int or size < 0 for size in shape)
        ):
            raise CheckpointSecurityError("safetensors tensor shape violates its budget")
        elements = math.prod(shape)
        if elements > MAX_TENSOR_ELEMENTS:
            raise CheckpointSecurityError("safetensors tensor element budget exceeded")
        offsets = item["data_offsets"]
        if (
            type(offsets) is not list
            or len(offsets) != 2
            or any(type(offset) is not int or offset < 0 for offset in offsets)
            or offsets[0] > offsets[1]
        ):
            raise CheckpointSecurityError("safetensors tensor offsets are invalid")
        tensor_bytes = offsets[1] - offsets[0]
        if tensor_bytes != elements * _DTYPE_BYTES[dtype]:
            raise CheckpointSecurityError("safetensors tensor byte span is inconsistent")
        if tensor_bytes > MAX_TENSOR_BYTES or offsets[1] > body_size:
            raise CheckpointSecurityError("safetensors tensor byte budget exceeded")
        total_elements += elements
        total_bytes += tensor_bytes
        if total_elements > MAX_TOTAL_ELEMENTS or total_bytes > MAX_TOTAL_TENSOR_BYTES:
            raise CheckpointSecurityError("safetensors aggregate tensor budget exceeded")
        ranges.append((offsets[0], offsets[1], key))
        normalized[key] = {"dtype": dtype, "shape": tuple(shape)}

    cursor = 0
    for start, end, _ in sorted(ranges):
        if start != cursor:
            raise CheckpointSecurityError("safetensors data ranges are not contiguous")
        cursor = end
    if cursor != body_size:
        raise CheckpointSecurityError("safetensors data region has trailing bytes")
    return normalized


def _expected_schema(
    config: Mapping[str, Any],
    has_ema: bool,
) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Derive the exact U-Net state contract without constructing a model."""
    model = config["model"]
    image_channels = model["image_channels"]
    base_channels = model["base_channels"]
    channel_mults = model["channel_mults"]
    num_res_blocks = model["num_res_blocks"]
    attention_resolutions = set(model["attention_resolutions"])
    time_embed_dim = model["time_embed_dim"]
    image_size = model["image_size"]
    unprefixed: dict[str, tuple[str, tuple[int, ...]]] = {}

    def add(name: str, *shape: int) -> None:
        if name in unprefixed:
            raise CheckpointSecurityError("model schema contains a duplicate key")
        unprefixed[name] = ("F32", tuple(shape))

    def add_linear(prefix: str, in_features: int, out_features: int) -> None:
        add(f"{prefix}.weight", out_features, in_features)
        add(f"{prefix}.bias", out_features)

    def add_conv(prefix: str, in_channels: int, out_channels: int, kernel: int) -> None:
        add(f"{prefix}.weight", out_channels, in_channels, kernel, kernel)
        add(f"{prefix}.bias", out_channels)

    def add_conv1d(prefix: str, in_channels: int, out_channels: int) -> None:
        add(f"{prefix}.weight", out_channels, in_channels, 1)
        add(f"{prefix}.bias", out_channels)

    def add_norm(prefix: str, channels: int) -> None:
        add(f"{prefix}.weight", channels)
        add(f"{prefix}.bias", channels)

    def add_residual(prefix: str, in_channels: int, out_channels: int) -> None:
        add_norm(f"{prefix}.norm1", in_channels)
        add_conv(f"{prefix}.conv1", in_channels, out_channels, 3)
        add_linear(f"{prefix}.time_proj", time_embed_dim, out_channels)
        add_norm(f"{prefix}.norm2", out_channels)
        add_conv(f"{prefix}.conv2", out_channels, out_channels, 3)
        if in_channels != out_channels:
            add_conv(f"{prefix}.skip", in_channels, out_channels, 1)

    def add_attention(prefix: str, channels: int) -> None:
        add_norm(f"{prefix}.norm", channels)
        add_conv1d(f"{prefix}.qkv", channels, channels * 3)
        add_conv1d(f"{prefix}.proj", channels, channels)

    add_linear("time_embed.1", base_channels, time_embed_dim)
    add_linear("time_embed.3", time_embed_dim, time_embed_dim)
    add_conv("input_conv", image_channels, base_channels, 3)

    channels = [base_channels]
    current_resolution = image_size
    in_channels = base_channels
    down_block_index = 0
    for level, multiplier in enumerate(channel_mults):
        out_channels = base_channels * multiplier
        for _ in range(num_res_blocks):
            prefix = f"down_blocks.{down_block_index}"
            add_residual(f"{prefix}.0", in_channels, out_channels)
            if current_resolution in attention_resolutions:
                add_attention(f"{prefix}.1", out_channels)
            channels.append(out_channels)
            in_channels = out_channels
            down_block_index += 1
        if level < len(channel_mults) - 1:
            add_conv(f"down_samples.{level}.conv", out_channels, out_channels, 3)
            channels.append(out_channels)
            current_resolution //= 2

    add_residual("mid_block1", in_channels, in_channels)
    add_attention("mid_attn", in_channels)
    add_residual("mid_block2", in_channels, in_channels)

    up_block_index = 0
    up_sample_index = 0
    for level in reversed(range(len(channel_mults))):
        out_channels = base_channels * channel_mults[level]
        for _ in range(num_res_blocks + 1):
            skip_channels = channels.pop()
            prefix = f"up_blocks.{up_block_index}"
            add_residual(f"{prefix}.0", in_channels + skip_channels, out_channels)
            if current_resolution in attention_resolutions:
                add_attention(f"{prefix}.1", out_channels)
            in_channels = out_channels
            up_block_index += 1
        if level > 0:
            add_conv(f"up_samples.{up_sample_index}.conv", out_channels, out_channels, 3)
            current_resolution *= 2
        up_sample_index += 1

    if channels:
        raise CheckpointSecurityError("model schema channel stack is inconsistent")
    add_norm("output_norm", in_channels)
    add_conv("output_conv", in_channels, image_channels, 3)

    schema: dict[str, tuple[str, tuple[int, ...]]] = {}
    for key, item in unprefixed.items():
        schema[f"model.{key}"] = item
        if has_ema:
            schema[f"ema.{key}"] = item
    if len(schema) > MAX_TENSORS:
        raise CheckpointSecurityError("model schema exceeds the tensor-count budget")
    return schema


def _validate_source_tensor(
    tensor: Any,
    *,
    expected_dtype: str,
    expected_shape: tuple[int, ...],
    label: str,
) -> tuple[int, int]:
    """Validate a live tensor before any CPU copy or serialization occurs."""
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device.type == "meta"
        or tensor.layout != torch.strided
        or _TORCH_DTYPE_TO_SAFE.get(tensor.dtype) != expected_dtype
        or tuple(tensor.shape) != expected_shape
    ):
        raise CheckpointSecurityError(f"{label} schema does not exactly match")
    elements = tensor.numel()
    tensor_bytes = elements * tensor.element_size()
    if elements > MAX_TENSOR_ELEMENTS or tensor_bytes > MAX_TENSOR_BYTES:
        raise CheckpointSecurityError(f"{label} exceeds its resource budget")
    return elements, tensor_bytes


def _validate_save_state(
    model_state: Mapping[str, torch.Tensor],
    ema_shadow: Mapping[str, torch.Tensor] | None,
    expected: Mapping[str, tuple[str, tuple[int, ...]]],
    parameter_keys: set[str],
) -> None:
    """Validate exact save-side keys, shapes, dtypes, and aggregate budgets."""
    expected_model = {
        key.removeprefix("model."): item
        for key, item in expected.items()
        if key.startswith("model.")
    }
    if set(model_state) != set(expected_model):
        raise CheckpointSecurityError("model state schema keys do not exactly match")
    if ema_shadow is not None:
        if len(ema_shadow) > MAX_TENSORS or set(ema_shadow) != parameter_keys:
            raise CheckpointSecurityError(
                "EMA keys do not exactly match trainable parameters"
            )

    total_elements = 0
    total_bytes = 0
    for key, (dtype, shape) in expected_model.items():
        elements, tensor_bytes = _validate_source_tensor(
            model_state[key],
            expected_dtype=dtype,
            expected_shape=shape,
            label="model state",
        )
        total_elements += elements
        total_bytes += tensor_bytes
        if ema_shadow is not None:
            ema_value = ema_shadow[key] if key in ema_shadow else model_state[key]
            elements, tensor_bytes = _validate_source_tensor(
                ema_value,
                expected_dtype=dtype,
                expected_shape=shape,
                label="EMA tensor",
            )
            total_elements += elements
            total_bytes += tensor_bytes
        if (
            total_elements > MAX_TOTAL_ELEMENTS
            or total_bytes > MAX_TOTAL_TENSOR_BYTES
        ):
            raise CheckpointSecurityError(
                "checkpoint save tensors exceed the aggregate resource budget"
            )


def load_checkpoint(
    weights_path: str | Path,
    *,
    device: torch.device,
    use_ema: bool = True,
) -> tuple[UNet, dict[str, Any]]:
    """Validate disk/header/CPU state before moving the model to the target device."""
    weights_path = Path(weights_path)
    if weights_path.suffix != ".safetensors":
        raise CheckpointSecurityError("checkpoint weights must use .safetensors")
    metadata_path = weights_path.with_suffix(".json")

    metadata = _parse_metadata(metadata_path)
    header = _inspect_header(weights_path)
    expected = _expected_schema(metadata["config"], metadata["has_ema"])
    if set(header) != set(expected):
        raise CheckpointSecurityError("checkpoint tensor keys do not exactly match the model")
    for key, (dtype, shape) in expected.items():
        if header[key]["dtype"] != dtype or header[key]["shape"] != shape:
            raise CheckpointSecurityError("checkpoint tensor schema does not exactly match")

    prefix = "ema." if use_ema and metadata["has_ema"] else "model."
    state: dict[str, torch.Tensor] = {}
    materialized_elements = 0
    materialized_bytes = 0
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(expected):
            raise CheckpointSecurityError("opened checkpoint keys changed after inspection")
        for key, (dtype, shape) in expected.items():
            tensor_slice = handle.get_slice(key)
            if tuple(tensor_slice.get_shape()) != shape:
                raise CheckpointSecurityError("opened checkpoint shape changed after inspection")
        for key, (dtype, shape) in expected.items():
            if not key.startswith(prefix):
                continue
            tensor = handle.get_tensor(key)
            if (
                tensor.device.type != "cpu"
                or tensor.layout != torch.strided
                or not tensor.is_contiguous()
                or _TORCH_DTYPE_TO_SAFE.get(tensor.dtype) != dtype
                or tuple(tensor.shape) != shape
                or tensor.numel() > MAX_TENSOR_ELEMENTS
                or tensor.numel() * tensor.element_size() > MAX_TENSOR_BYTES
            ):
                raise CheckpointSecurityError("materialized checkpoint tensor is invalid")
            materialized_elements += tensor.numel()
            materialized_bytes += tensor.numel() * tensor.element_size()
            if (
                materialized_elements > MAX_TOTAL_ELEMENTS
                or materialized_bytes > MAX_TOTAL_TENSOR_BYTES
            ):
                raise CheckpointSecurityError("materialized checkpoint aggregate is excessive")
            state[key[len(prefix):]] = tensor
    try:
        model = UNet(_unet_config(metadata["config"]))
        model.load_state_dict(state, strict=True)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise CheckpointSecurityError("checkpoint state failed strict CPU loading") from exc
    model = model.to(device)
    model.eval()
    return model, metadata


def _publish_new_file(temp_path: Path, destination: Path) -> None:
    try:
        os.link(temp_path, destination)
    except FileExistsError as exc:
        raise CheckpointSecurityError(f"refusing to overwrite {destination}") from exc
    temp_path.unlink()


def save_checkpoint(
    weights_path: str | Path,
    *,
    model: UNet,
    ema_shadow: Mapping[str, torch.Tensor] | None,
    config: Mapping[str, Any],
    step: int,
    loss: float | None = None,
) -> tuple[Path, Path]:
    """Write a new safetensors/JSON pair without serializing optimizer objects."""
    weights_path = Path(weights_path)
    if weights_path.suffix != ".safetensors":
        raise CheckpointSecurityError("checkpoint weights must use .safetensors")
    metadata_path = weights_path.with_suffix(".json")
    normalized_config = validate_config(dict(config))
    normalized_step = _integer(step, "checkpoint step", 0, 100_000_000)
    normalized_loss = (
        None if loss is None else _number(loss, "checkpoint loss", -1e12, 1e12)
    )

    model_state = model.state_dict()
    parameter_keys = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    expected = _expected_schema(normalized_config, ema_shadow is not None)
    _validate_save_state(model_state, ema_shadow, expected, parameter_keys)

    directory = weights_path.parent
    _reject_linked_components(directory)
    if not directory.exists():
        directory.mkdir(parents=True, mode=0o700)
    _reject_linked_components(directory)
    if not directory.is_dir():
        raise CheckpointSecurityError("checkpoint parent must be a directory")
    if os.name != "nt":
        os.chmod(directory, 0o700)
    if os.path.lexists(weights_path) or os.path.lexists(metadata_path):
        raise CheckpointSecurityError("refusing to overwrite an existing checkpoint pair")

    tensors: dict[str, torch.Tensor] = {}
    for key, value in model_state.items():
        tensor = value.detach().to(device="cpu").contiguous().clone()
        tensors[f"model.{key}"] = tensor
        if ema_shadow is not None:
            ema_value = ema_shadow[key] if key in ema_shadow else value
            tensors[f"ema.{key}"] = (
                ema_value.detach().to(device="cpu").contiguous().clone()
            )

    metadata = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "step": normalized_step,
        "loss": normalized_loss,
        "has_ema": ema_shadow is not None,
        "config": normalized_config,
    }
    metadata_bytes = (
        json.dumps(
            metadata,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if len(metadata_bytes) > MAX_METADATA_BYTES:
        raise CheckpointSecurityError("checkpoint metadata exceeds its size limit")

    weights_temp = directory / f".{weights_path.name}.{secrets.token_hex(12)}.tmp"
    metadata_temp = directory / f".{metadata_path.name}.{secrets.token_hex(12)}.tmp"
    try:
        save_file(tensors, weights_temp)
        if weights_temp.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise CheckpointSecurityError("generated checkpoint exceeds its file limit")
        if os.name != "nt":
            os.chmod(weights_temp, 0o600)
        # Windows rejects FlushFileBuffers on a read-only descriptor.
        with weights_temp.open("rb+") as handle:
            os.fsync(handle.fileno())

        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(metadata_temp, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(metadata_bytes)
            handle.flush()
            os.fsync(handle.fileno())

        _publish_new_file(weights_temp, weights_path)
        try:
            _publish_new_file(metadata_temp, metadata_path)
        except Exception:
            # An orphaned tensor file is inert, but remove it for pair atomicity.
            weights_path.unlink(missing_ok=True)
            raise
    finally:
        weights_temp.unlink(missing_ok=True)
        metadata_temp.unlink(missing_ok=True)
    return weights_path, metadata_path

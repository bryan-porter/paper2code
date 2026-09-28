from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

ROOT = Path(__file__).parents[1]
DDPM_SRC = ROOT / "skills" / "paper2code" / "worked" / "ddpm" / "src"
sys.path.insert(0, str(DDPM_SRC))

import checkpoint  # noqa: E402
from model import UNet, UNetConfig  # noqa: E402


def tiny_config() -> dict:
    return {
        "diffusion": {
            "timesteps": 8,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "schedule": "linear",
            "variance_type": "fixed_small",
        },
        "model": {
            "image_channels": 1,
            "base_channels": 8,
            "channel_mults": [1],
            "num_res_blocks": 1,
            "attention_resolutions": [],
            "dropout": 0.0,
            "time_embed_dim": 8,
            "num_groups": 1,
            "image_size": 8,
        },
        "training": {
            "optimizer": "adam",
            "lr": 0.0002,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "total_steps": 10,
            "batch_size": 2,
            "gradient_clip": 1.0,
            "schedule": "constant",
            "ema_decay": 0.999,
            "ema_start_step": 0,
        },
        "data": {
            "dataset": "fixture",
            "image_size": 8,
            "augmentation": "none",
            "normalize": True,
        },
        "eval": {"metric": "FID", "num_samples": 2, "sampling_steps": 8},
    }


def tiny_model() -> UNet:
    model = tiny_config()["model"]
    return UNet(
        UNetConfig(
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
    )


def save_valid_pair(tmp_path: Path) -> tuple[Path, UNet, dict[str, torch.Tensor]]:
    model = tiny_model()
    ema = {
        name: parameter.detach().clone().add(0.25)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    path = tmp_path / "checkpoint.safetensors"
    checkpoint.save_checkpoint(
        path,
        model=model,
        ema_shadow=ema,
        config=tiny_config(),
        step=4,
        loss=0.5,
    )
    return path, model, ema


def test_safetensors_pair_round_trips_ema_on_cpu(tmp_path: Path) -> None:
    path, _, ema = save_valid_pair(tmp_path)

    loaded, metadata = checkpoint.load_checkpoint(
        path,
        device=torch.device("cpu"),
        use_ema=True,
    )

    assert path.is_file()
    assert path.with_suffix(".json").is_file()
    assert metadata["format"] == checkpoint.FORMAT_NAME
    assert metadata["step"] == 4
    for name, parameter in loaded.named_parameters():
        assert torch.equal(parameter, ema[name])


def test_checkpoint_without_ema_round_trips_model_state(tmp_path: Path) -> None:
    model = tiny_model()
    path = tmp_path / "checkpoint.safetensors"
    checkpoint.save_checkpoint(
        path,
        model=model,
        ema_shadow=None,
        config=tiny_config(),
        step=1,
    )

    loaded, metadata = checkpoint.load_checkpoint(
        path,
        device=torch.device("cpu"),
    )

    assert metadata["has_ema"] is False
    for name, expected in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], expected)


def test_checkpoint_rejects_ema_key_mismatch_before_writing(tmp_path: Path) -> None:
    model = tiny_model()

    with pytest.raises(checkpoint.CheckpointSecurityError, match="EMA keys"):
        checkpoint.save_checkpoint(
            tmp_path / "checkpoint.safetensors",
            model=model,
            ema_shadow={"unexpected": torch.zeros(1)},
            config=tiny_config(),
            step=1,
        )

    assert list(tmp_path.iterdir()) == []


def test_save_rejects_model_config_schema_mismatch_before_writing(
    tmp_path: Path,
) -> None:
    model = tiny_model()
    config = tiny_config()
    config["model"]["image_channels"] = 2

    with pytest.raises(checkpoint.CheckpointSecurityError, match="model state schema"):
        checkpoint.save_checkpoint(
            tmp_path / "checkpoint.safetensors",
            model=model,
            ema_shadow=None,
            config=config,
            step=1,
        )

    assert list(tmp_path.iterdir()) == []


def test_save_rejects_ema_tensor_schema_mismatch_before_writing(
    tmp_path: Path,
) -> None:
    model = tiny_model()
    ema = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    first_key = next(iter(ema))
    ema[first_key] = torch.zeros(1, dtype=torch.float32)

    with pytest.raises(checkpoint.CheckpointSecurityError, match="EMA tensor schema"):
        checkpoint.save_checkpoint(
            tmp_path / "checkpoint.safetensors",
            model=model,
            ema_shadow=ema,
            config=tiny_config(),
            step=1,
        )

    assert list(tmp_path.iterdir()) == []


def test_checkpoint_rejects_out_of_range_config_before_schema_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _, _ = save_valid_pair(tmp_path)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["config"]["model"]["base_channels"] = 1_000_000
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    def must_not_construct(*args, **kwargs):
        raise AssertionError("UNet construction must not happen for invalid config")

    monkeypatch.setattr(checkpoint, "UNet", must_not_construct)
    with pytest.raises(checkpoint.CheckpointSecurityError, match="base_channels"):
        checkpoint.load_checkpoint(path, device=torch.device("cpu"))


def test_checkpoint_rejects_unexpected_tensor_key(tmp_path: Path) -> None:
    valid_path, _, _ = save_valid_pair(tmp_path)
    forged_path = tmp_path / "forged.safetensors"
    from safetensors.torch import save_file

    save_file({"unexpected": torch.zeros(1)}, forged_path)
    shutil.copyfile(valid_path.with_suffix(".json"), forged_path.with_suffix(".json"))

    with pytest.raises(checkpoint.CheckpointSecurityError, match="keys"):
        checkpoint.load_checkpoint(forged_path, device=torch.device("cpu"))


def test_checkpoint_file_limit_is_checked_before_safetensors_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _, _ = save_valid_pair(tmp_path)
    monkeypatch.setattr(checkpoint, "MAX_CHECKPOINT_BYTES", path.stat().st_size - 1)

    def must_not_open(*args, **kwargs):
        raise AssertionError("safetensors must not open an oversized file")

    monkeypatch.setattr(checkpoint, "safe_open", must_not_open)
    with pytest.raises(checkpoint.CheckpointSecurityError, match="file-size"):
        checkpoint.load_checkpoint(path, device=torch.device("cpu"))


def test_checkpoint_rejects_unknown_metadata_field(tmp_path: Path) -> None:
    path, _, _ = save_valid_pair(tmp_path)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["unexpected"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(checkpoint.CheckpointSecurityError, match="strict schema"):
        checkpoint.load_checkpoint(path, device=torch.device("cpu"))


def test_expected_schema_is_derived_without_constructing_unet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model()
    expected = {}
    for name, tensor in model.state_dict().items():
        expected[f"model.{name}"] = ("F32", tuple(tensor.shape))
        expected[f"ema.{name}"] = ("F32", tuple(tensor.shape))

    def must_not_construct(*args, **kwargs):
        raise AssertionError("schema validation must precede UNet construction")

    monkeypatch.setattr(checkpoint, "UNet", must_not_construct)
    assert checkpoint._expected_schema(tiny_config(), has_ema=True) == expected


def test_static_schema_matches_multilevel_attention_unet() -> None:
    config = tiny_config()
    config["model"]["channel_mults"] = [1, 2]
    config["model"]["attention_resolutions"] = [4]
    model = UNet(checkpoint._unet_config(checkpoint.validate_config(config)))
    expected = {
        f"model.{name}": ("F32", tuple(tensor.shape))
        for name, tensor in model.state_dict().items()
    }

    assert checkpoint._expected_schema(
        checkpoint.validate_config(config),
        has_ema=False,
    ) == expected


def test_loader_materializes_only_selected_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _, _ = save_valid_pair(tmp_path)
    original_safe_open = checkpoint.safe_open
    materialized: list[str] = []

    class TrackedSafeOpen:
        def __init__(self, *args, **kwargs):
            self._context = original_safe_open(*args, **kwargs)
            self._handle = None

        def __enter__(self):
            self._handle = self._context.__enter__()
            return self

        def __exit__(self, *args):
            return self._context.__exit__(*args)

        def keys(self):
            return self._handle.keys()

        def get_slice(self, key):
            return self._handle.get_slice(key)

        def get_tensor(self, key):
            materialized.append(key)
            return self._handle.get_tensor(key)

    monkeypatch.setattr(checkpoint, "safe_open", TrackedSafeOpen)
    checkpoint.load_checkpoint(path, device=torch.device("cpu"), use_ema=False)

    assert materialized
    assert all(key.startswith("model.") for key in materialized)


@pytest.mark.parametrize(
    ("limit_name", "limit_value", "message"),
    [
        ("MAX_HEADER_BYTES", 1, "header"),
        ("MAX_TENSORS", 1, "tensor-count"),
        ("MAX_TENSOR_RANK", 0, "shape"),
        ("MAX_TENSOR_ELEMENTS", 0, "element"),
        ("MAX_TOTAL_ELEMENTS", 1, "aggregate"),
        ("MAX_TENSOR_BYTES", 0, "byte"),
        ("MAX_TOTAL_TENSOR_BYTES", 1, "aggregate"),
    ],
)
def test_checkpoint_header_resource_budgets_precede_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit_value: int,
    message: str,
) -> None:
    path, _, _ = save_valid_pair(tmp_path)
    monkeypatch.setattr(checkpoint, limit_name, limit_value)

    def must_not_open(*args, **kwargs):
        raise AssertionError("tensor materialization must follow header budgets")

    monkeypatch.setattr(checkpoint, "safe_open", must_not_open)
    with pytest.raises(checkpoint.CheckpointSecurityError, match=message):
        checkpoint.load_checkpoint(path, device=torch.device("cpu"))


def test_checkpoint_rejects_unsupported_tensor_dtype(tmp_path: Path) -> None:
    valid_path, _, _ = save_valid_pair(tmp_path)
    forged_path = tmp_path / "forged.safetensors"
    from safetensors.torch import save_file

    save_file({"model.untrusted": torch.zeros(1, dtype=torch.float64)}, forged_path)
    shutil.copyfile(valid_path.with_suffix(".json"), forged_path.with_suffix(".json"))

    with pytest.raises(checkpoint.CheckpointSecurityError, match="dtype"):
        checkpoint.load_checkpoint(forged_path, device=torch.device("cpu"))


def test_checkpoint_rejects_exact_key_with_wrong_shape(tmp_path: Path) -> None:
    model = tiny_model()
    valid_path = tmp_path / "valid.safetensors"
    checkpoint.save_checkpoint(
        valid_path,
        model=model,
        ema_shadow=None,
        config=tiny_config(),
        step=1,
    )
    tensors = {
        f"model.{name}": value.detach().clone()
        for name, value in model.state_dict().items()
    }
    first_key = next(iter(tensors))
    tensors[first_key] = tensors[first_key].reshape(-1)
    forged_path = tmp_path / "forged.safetensors"
    from safetensors.torch import save_file

    save_file(tensors, forged_path)
    shutil.copyfile(valid_path.with_suffix(".json"), forged_path.with_suffix(".json"))

    with pytest.raises(checkpoint.CheckpointSecurityError, match="schema"):
        checkpoint.load_checkpoint(forged_path, device=torch.device("cpu"))


def test_checkpoint_rejects_non_regular_weights_path(tmp_path: Path) -> None:
    path, _, _ = save_valid_pair(tmp_path)
    path.unlink()
    path.mkdir()

    with pytest.raises(checkpoint.CheckpointSecurityError, match="regular"):
        checkpoint.load_checkpoint(path, device=torch.device("cpu"))

from __future__ import annotations

import importlib.util
import hashlib
import io
import os
import stat
import struct
import sys
import tempfile
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


DDPM_SRC = Path(__file__).parents[1] / "skills" / "paper2code" / "worked" / "ddpm" / "src"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_evaluate(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    torch = ModuleType("torch")
    torch.Tensor = object
    torch.device = str
    torch.no_grad = lambda: (lambda function: function)
    torchvision = ModuleType("torchvision")
    model = ModuleType("model")
    model.UNet = object
    utils = ModuleType("utils")
    utils.linear_noise_schedule = lambda *args: {}
    utils.sample = lambda *args: None
    checkpoint = ModuleType("checkpoint")
    checkpoint.load_checkpoint = lambda *args, **kwargs: None
    for name, module in (
        ("torch", torch),
        ("torchvision", torchvision),
        ("model", model),
        ("utils", utils),
        ("checkpoint", checkpoint),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return _load_module("ddpm_evaluate_offline_test", DDPM_SRC / "evaluate.py")


def test_fid_requires_local_safe_weights_before_entering_upstream_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluate = _load_evaluate(monkeypatch)
    upstream = ModuleType("pytorch_fid")
    upstream.fid_score = SimpleNamespace(
        calculate_fid_given_paths=lambda *args, **kwargs: pytest.fail(
            "upstream FID constructor can auto-download pickle weights"
        )
    )
    monkeypatch.setitem(sys.modules, "pytorch_fid", upstream)

    with pytest.raises(ValueError, match="local.*safetensors"):
        evaluate.compute_fid(str(DDPM_SRC), str(Path(__file__)), device="cpu")


def test_fid_uses_local_weights_and_restores_upstream_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluate = _load_evaluate(monkeypatch)
    monkeypatch.setattr(evaluate, "version", lambda name: "0.3.0")
    weights = {"known": object()}
    monkeypatch.setattr(evaluate, "_load_local_fid_weights", lambda path: weights)
    calls: list[str] = []
    preflight: list[tuple[str, int]] = []
    monkeypatch.setattr(
        evaluate,
        "_preflight_fid_path",
        lambda path, dims: preflight.append((path, dims)),
        raising=False,
    )

    def forbidden_remote_loader(url: str):
        pytest.fail(f"unexpected remote model load: {url}")

    inception = ModuleType("pytorch_fid.inception")
    inception.FID_WEIGHTS_URL = evaluate.FID_WEIGHTS_URL
    inception.load_state_dict_from_url = forbidden_remote_loader

    class FakeInception:
        BLOCK_INDEX_BY_DIM = {2048: 3}

        def __init__(self, blocks: list[int]) -> None:
            assert blocks == [3]
            assert inception.load_state_dict_from_url(evaluate.FID_WEIGHTS_URL) is weights

        def to(self, device: str):
            assert device == "cpu"
            return self

    inception.InceptionV3 = FakeInception
    fid_score = ModuleType("pytorch_fid.fid_score")

    def statistics(path, model, batch_size, dims, device):
        assert isinstance(model, FakeInception)
        assert (batch_size, dims, device) == (50, 2048, "cpu")
        calls.append(path)
        return (path, path)

    fid_score.compute_statistics_of_path = statistics
    fid_score.calculate_frechet_distance = lambda *args: 3.5
    upstream = ModuleType("pytorch_fid")
    upstream.fid_score = fid_score
    upstream.inception = inception
    monkeypatch.setitem(sys.modules, "pytorch_fid", upstream)
    monkeypatch.setitem(sys.modules, "pytorch_fid.fid_score", fid_score)
    monkeypatch.setitem(sys.modules, "pytorch_fid.inception", inception)

    generated_path = str(DDPM_SRC)
    real_path = str(Path(__file__))
    assert evaluate.compute_fid(
        generated_path,
        real_path,
        device="cpu",
        inception_weights_path="local.safetensors",
    ) == 3.5
    assert calls == [generated_path, real_path]
    assert preflight == [(generated_path, 2048), (real_path, 2048)]
    assert inception.load_state_dict_from_url is forbidden_remote_loader


def test_fid_rejects_large_stats_archive_before_numpy_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    numpy = pytest.importorskip("numpy")
    evaluate = _load_evaluate(monkeypatch)
    monkeypatch.setattr(evaluate, "MAX_FID_STATS_ARCHIVE_BYTES", 100, raising=False)
    with tempfile.NamedTemporaryFile(
        dir=Path(__file__).parent, suffix=".npz", delete=False
    ) as handle:
        path = Path(handle.name)
    try:
        numpy.savez(path, mu=numpy.zeros(4), sigma=numpy.eye(4))
        with pytest.raises(ValueError, match="archive.*size"):
            evaluate._preflight_fid_path(str(path), 4)
    finally:
        path.unlink(missing_ok=True)


def test_fid_rejects_stats_shape_before_loading_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    numpy = pytest.importorskip("numpy")
    evaluate = _load_evaluate(monkeypatch)
    with tempfile.NamedTemporaryFile(
        dir=Path(__file__).parent, suffix=".npz", delete=False
    ) as handle:
        path = Path(handle.name)
    try:
        numpy.savez_compressed(path, mu=numpy.zeros(4), sigma=numpy.eye(8))
        with pytest.raises(ValueError, match="shape"):
            evaluate._preflight_fid_path(str(path), 4)
    finally:
        path.unlink(missing_ok=True)


def test_fid_rejects_compressed_member_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    numpy = pytest.importorskip("numpy")
    evaluate = _load_evaluate(monkeypatch)
    monkeypatch.setattr(evaluate, "MAX_FID_STATS_MEMBER_BYTES", 1024)
    with tempfile.NamedTemporaryFile(
        dir=Path(__file__).parent, suffix=".npz", delete=False
    ) as handle:
        path = Path(handle.name)
    try:
        numpy.savez_compressed(path, mu=numpy.zeros(4), sigma=numpy.zeros((100, 100)))
        assert path.stat().st_size < 1024
        with pytest.raises(ValueError, match="unsupported array"):
            evaluate._preflight_fid_path(str(path), 4)
    finally:
        path.unlink(missing_ok=True)


def test_fid_rejects_declared_large_header_before_numpy_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    numpy = pytest.importorskip("numpy")
    evaluate = _load_evaluate(monkeypatch)
    with tempfile.NamedTemporaryFile(
        dir=Path(__file__).parent, suffix=".npz", delete=False
    ) as handle:
        path = Path(handle.name)
    try:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("mu.npy", b"\x93NUMPY\x02\x00" + struct.pack("<I", 1_000_000))
            archive.writestr("sigma.npy", b"\x93NUMPY\x02\x00" + struct.pack("<I", 1_000_000))
        monkeypatch.setattr(
            numpy.lib.format,
            "read_array_header_2_0",
            lambda *args, **kwargs: pytest.fail("oversized header reached NumPy parser"),
        )
        with pytest.raises(ValueError, match="header exceeds"):
            evaluate._preflight_fid_path(str(path), 4)
    finally:
        path.unlink(missing_ok=True)


def test_fid_rejects_zip64_end_record_before_zipfile_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    numpy = pytest.importorskip("numpy")
    evaluate = _load_evaluate(monkeypatch)
    standard = io.BytesIO()
    numpy.savez_compressed(standard, mu=numpy.zeros(4), sigma=numpy.eye(4))
    raw = standard.getvalue()
    end_offset = raw.rfind(b"PK\x05\x06")
    assert end_offset + 22 == len(raw)
    _, _, _, _, _, directory_size, directory_offset, _ = struct.unpack_from(
        "<4s4H2LH", raw, end_offset
    )
    zip64_end = struct.pack(
        "<4sQ2H2L4Q",
        b"PK\x06\x06", 44, 45, 45, 0, 0,
        100_000, 100_000, directory_size, directory_offset,
    )
    zip64_locator = struct.pack("<4sLQL", b"PK\x06\x07", 0, end_offset, 1)
    forged = raw[:end_offset] + zip64_end + zip64_locator + raw[end_offset:]
    with zipfile.ZipFile(io.BytesIO(forged)) as archive:
        assert set(archive.namelist()) == {"mu.npy", "sigma.npy"}
    monkeypatch.setattr(
        evaluate.zipfile,
        "ZipFile",
        lambda *args, **kwargs: pytest.fail("ZIP64 reached the ZIP parser"),
    )

    with tempfile.NamedTemporaryFile(
        dir=Path(__file__).parent, suffix=".npz", delete=False
    ) as handle:
        path = Path(handle.name)
        handle.write(forged)
    try:
        with pytest.raises(ValueError, match="ZIP64"):
            evaluate._preflight_fid_path(str(path), 4)
    finally:
        path.unlink(missing_ok=True)


def test_fid_rejects_classic_zip64_sentinel_before_zipfile_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    numpy = pytest.importorskip("numpy")
    evaluate = _load_evaluate(monkeypatch)
    standard = io.BytesIO()
    numpy.savez_compressed(standard, mu=numpy.zeros(4), sigma=numpy.eye(4))
    forged = bytearray(standard.getvalue())
    end_offset = forged.rfind(b"PK\x05\x06")
    struct.pack_into("<H", forged, end_offset + 10, 0xFFFF)
    monkeypatch.setattr(
        evaluate.zipfile,
        "ZipFile",
        lambda *args, **kwargs: pytest.fail("ZIP64 reached the ZIP parser"),
    )
    with tempfile.NamedTemporaryFile(
        dir=Path(__file__).parent, suffix=".npz", delete=False
    ) as handle:
        path = Path(handle.name)
        handle.write(forged)
    try:
        with pytest.raises(ValueError, match="ZIP64"):
            evaluate._preflight_fid_path(str(path), 4)
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "path",
    [
        r"\\server\share\stats.npz",
        r"//server/share/stats.npz",
        r"\\?\C:\stats.npz",
        r"\\.\pipe\fid",
    ],
)
def test_fid_rejects_windows_unc_and_device_paths_before_access(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    if os.name != "nt":
        pytest.skip("Windows path namespace boundary")
    evaluate = _load_evaluate(monkeypatch)
    with pytest.raises(ValueError, match="UNC|device"):
        evaluate._checked_local_fid_path(path)


@pytest.mark.parametrize("drive_type", [0, 1, 4])
def test_fid_rejects_remote_or_unresolved_drive_before_lstat(
    monkeypatch: pytest.MonkeyPatch, drive_type: int
) -> None:
    if os.name != "nt":
        pytest.skip("Windows drive-root boundary")
    evaluate = _load_evaluate(monkeypatch)
    monkeypatch.setattr(
        evaluate, "_windows_fid_drive_type", lambda root: drive_type, raising=False
    )
    monkeypatch.setattr(
        Path, "lstat", lambda self: pytest.fail("drive path was stat'ed before classification")
    )
    with pytest.raises(ValueError, match="remote|unknown|invalid"):
        evaluate._checked_local_fid_path(str(Path(__file__)))


def test_fid_rejects_relative_path_under_remote_drive_cwd_before_lstat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows drive-root boundary")
    evaluate = _load_evaluate(monkeypatch)
    monkeypatch.setattr(evaluate.os.path, "abspath", lambda path: r"Z:\mapped\fid.npz")
    roots: list[str] = []
    monkeypatch.setattr(
        evaluate,
        "_windows_fid_drive_type",
        lambda root: roots.append(root) or 4,
        raising=False,
    )
    monkeypatch.setattr(
        Path, "lstat", lambda self: pytest.fail("relative path was stat'ed")
    )
    with pytest.raises(ValueError, match="remote"):
        evaluate._checked_local_fid_path("fid.npz")
    assert roots == ["Z:\\"]


def test_fid_windows_drive_type_api_accepts_workspace_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows drive-root boundary")
    evaluate = _load_evaluate(monkeypatch)
    assert evaluate._windows_fid_drive_type(Path.cwd().anchor) in {2, 3, 5, 6}


def test_fid_rejects_reparse_ancestor_for_stats_and_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluate = _load_evaluate(monkeypatch)
    ancestor = Path(__file__).parent
    real_lstat = Path.lstat

    def lstat_with_reparse(self):
        metadata = real_lstat(self)
        if self == ancestor:
            return SimpleNamespace(
                st_mode=stat.S_IFDIR,
                st_file_attributes=getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400),
            )
        return metadata

    monkeypatch.setattr(Path, "lstat", lstat_with_reparse)
    safetensors = ModuleType("safetensors")
    safetensors.safe_open = lambda *args, **kwargs: pytest.fail(
        "reparse ancestor reached the weights parser"
    )
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    with pytest.raises(ValueError, match="ancestor|reparse"):
        evaluate._preflight_fid_path(str(Path(__file__)), 4)
    with pytest.raises(ValueError, match="ancestor|reparse"):
        evaluate._load_local_fid_weights(str(Path(__file__)))


@pytest.mark.parametrize("invalid_input", ["generated", "real"])
def test_fid_rejects_bad_stats_on_either_path_before_model_load(
    monkeypatch: pytest.MonkeyPatch, invalid_input: str
) -> None:
    numpy = pytest.importorskip("numpy")
    evaluate = _load_evaluate(monkeypatch)
    monkeypatch.setattr(evaluate, "version", lambda name: "0.3.0")
    monkeypatch.setattr(
        evaluate,
        "_load_local_fid_weights",
        lambda path: pytest.fail("invalid FID input reached model load"),
    )
    paths = []
    try:
        for _ in range(2):
            with tempfile.NamedTemporaryFile(
                dir=Path(__file__).parent, suffix=".npz", delete=False
            ) as handle:
                paths.append(Path(handle.name))
        numpy.savez_compressed(paths[0], mu=numpy.zeros(4), sigma=numpy.eye(4))
        numpy.savez_compressed(paths[1], mu=numpy.zeros(4), sigma=numpy.eye(8))
        first, second = (paths[1], paths[0]) if invalid_input == "generated" else paths
        with pytest.raises(ValueError, match="shape"):
            evaluate.compute_fid(
                str(first), str(second), dims=4, inception_weights_path="local.safetensors"
            )
    finally:
        for path in paths:
            path.unlink(missing_ok=True)


def test_fid_accepts_standard_2048_dimensional_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    numpy = pytest.importorskip("numpy")
    evaluate = _load_evaluate(monkeypatch)
    with tempfile.NamedTemporaryFile(
        dir=Path(__file__).parent, suffix=".npz", delete=False
    ) as handle:
        path = Path(handle.name)
    try:
        numpy.savez_compressed(
            path,
            mu=numpy.zeros(2048, dtype=numpy.float64),
            sigma=numpy.zeros((2048, 2048), dtype=numpy.float64),
        )
        evaluate._preflight_fid_path(str(path), 2048)
    finally:
        path.unlink(missing_ok=True)


def test_fid_rejects_oversized_image_before_upstream_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_module = pytest.importorskip("PIL.Image")
    evaluate = _load_evaluate(monkeypatch)
    monkeypatch.setattr(evaluate, "MAX_FID_IMAGE_PIXELS", 32, raising=False)
    with tempfile.NamedTemporaryFile(
        dir=Path(__file__).parent, suffix=".png", delete=False
    ) as handle:
        path = Path(handle.name)
    try:
        image_module.new("RGB", (8, 8)).save(path)
        with pytest.raises(ValueError, match="pixels"):
            evaluate._preflight_fid_path(str(path.parent), 2048)
    finally:
        path.unlink(missing_ok=True)


def test_fid_accepts_normal_local_image_set(monkeypatch: pytest.MonkeyPatch) -> None:
    image_module = pytest.importorskip("PIL.Image")
    evaluate = _load_evaluate(monkeypatch)
    with tempfile.NamedTemporaryFile(
        dir=Path(__file__).parent, suffix=".png", delete=False
    ) as handle:
        path = Path(handle.name)
    try:
        image_module.new("RGB", (8, 8)).save(path)
        evaluate._preflight_fid_path(str(path.parent), 2048)
    finally:
        path.unlink(missing_ok=True)


def test_fid_rejects_oversized_local_weights_before_opening_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluate = _load_evaluate(monkeypatch)
    monkeypatch.setattr(evaluate, "MAX_FID_WEIGHTS_BYTES", 8)
    safetensors = ModuleType("safetensors")
    safetensors.safe_open = lambda *args, **kwargs: pytest.fail(
        "oversized weights must not reach the parser"
    )
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)

    with tempfile.NamedTemporaryFile(dir=Path(__file__).parent, delete=False) as handle:
        handle.write(b"123456789")
    try:
        with pytest.raises(ValueError, match="file-size"):
            evaluate._load_local_fid_weights(handle.name)
    finally:
        Path(handle.name).unlink()


def test_fid_converter_rejects_unverified_pickle_before_loading_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter = _load_module(
        "fid_converter_test", Path(__file__).parents[1] / "scripts" / "convert_fid_inception.py"
    )
    monkeypatch.setattr(converter, "SOURCE_BYTES", 4)
    with tempfile.NamedTemporaryFile(dir=Path(__file__).parent, delete=False) as handle:
        handle.write(b"test")
    try:
        with pytest.raises(ValueError, match="SHA-256"):
            converter.convert(
                Path(handle.name), Path(handle.name + ".safetensors")
            )
    finally:
        Path(handle.name).unlink()


def test_fid_converter_restricts_pickle_and_publishes_safe_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter = _load_module(
        "fid_converter_success_test",
        Path(__file__).parents[1] / "scripts" / "convert_fid_inception.py",
    )
    source_body = b"known source fixture"
    monkeypatch.setattr(converter, "SOURCE_BYTES", len(source_body))
    monkeypatch.setattr(converter, "SOURCE_SHA256", hashlib.sha256(source_body).hexdigest())
    torch = ModuleType("torch")
    torch.float32 = object()
    torch.int64 = object()
    torch.strided = object()

    class FakeTensor:
        layout = torch.strided
        dtype = torch.float32
        ndim = 1

        def numel(self) -> int:
            return 2

        def detach(self):
            return self

        def contiguous(self):
            return self

    torch.Tensor = FakeTensor

    def restricted_load(handle, *, map_location, weights_only):
        assert handle.read() == source_body
        assert map_location == "cpu"
        assert weights_only is True
        return {"inception.weight": FakeTensor()}

    torch.load = restricted_load
    safetensors = ModuleType("safetensors")
    safetensors.__path__ = []
    safetensors_torch = ModuleType("safetensors.torch")

    def save_file(tensors, path):
        assert list(tensors) == ["inception.weight"]
        Path(path).write_bytes(b"safe output")

    safetensors_torch.save_file = save_file
    for name, module in (
        ("torch", torch),
        ("safetensors", safetensors),
        ("safetensors.torch", safetensors_torch),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    with tempfile.NamedTemporaryFile(dir=Path(__file__).parent, delete=False) as handle:
        handle.write(source_body)
    source = Path(handle.name)
    destination = Path(handle.name + ".safetensors")
    try:
        converter.convert(source, destination)
        assert destination.read_bytes() == b"safe output"
        assert source.read_bytes() == source_body
    finally:
        source.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)


def test_cifar_loaders_never_download_implicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    torch = ModuleType("torch")
    torch.__path__ = []
    torch_utils = ModuleType("torch.utils")
    torch_utils.__path__ = []
    torch_data = ModuleType("torch.utils.data")
    torch_data.Dataset = object
    torch_data.DataLoader = lambda dataset, **kwargs: (dataset, kwargs)
    torchvision = ModuleType("torchvision")
    datasets = ModuleType("torchvision.datasets")

    def cifar10(**kwargs):
        calls.append(kwargs)
        return object()

    datasets.CIFAR10 = cifar10
    transforms = ModuleType("torchvision.transforms")
    transforms.Compose = lambda items: items
    transforms.RandomHorizontalFlip = lambda: object()
    transforms.ToTensor = lambda: object()
    transforms.Normalize = lambda **kwargs: object()
    torchvision.datasets = datasets
    torchvision.transforms = transforms
    for name, module in (
        ("torch", torch),
        ("torch.utils", torch_utils),
        ("torch.utils.data", torch_data),
        ("torchvision", torchvision),
        ("torchvision.datasets", datasets),
        ("torchvision.transforms", transforms),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    data = _load_module("ddpm_data_offline_test", DDPM_SRC / "data.py")
    data.get_dataloaders(data_dir="preverified-cifar")

    assert [call["train"] for call in calls] == [True, False]
    assert all(call["download"] is False for call in calls)

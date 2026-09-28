"""
Denoising Diffusion Probabilistic Models — Evaluation (FID Score)

Paper: https://arxiv.org/abs/2006.11239
Authors: Ho, Jain, Abbeel (2020)

§4 — "We report FID score and Inception Score... Our best results
are FID: 3.17, IS: 9.46 (unconditional) on CIFAR-10."

This module provides a wrapper for sampling from a trained DDPM model
and computing FID scores using the pytorch-fid library.

FID (Fréchet Inception Distance) is the primary evaluation metric
used in §4 and Table 1. Lower FID = better quality.

NOTE: FID computation requires:
  1. Generating 50K samples (standard for CIFAR-10)
  2. Computing Inception features for real and generated images
  3. Computing the Fréchet distance between the two feature distributions

This is computationally expensive. For quick validation, generate a
small batch and visually inspect.
"""

import io
import logging
import math
import os
import stat
import struct
import threading
import warnings
import zipfile
from importlib.metadata import version
from pathlib import Path
from typing import Optional

import torch
import torchvision
from model import UNet
from utils import linear_noise_schedule, sample
from checkpoint import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
logger = logging.getLogger(__name__)
FID_WEIGHTS_URL = (
    "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/"
    "pt_inception-2015-12-05-6726825d.pth"
)
MAX_FID_WEIGHTS_BYTES = 128 * 1024 * 1024
MAX_FID_TENSORS = 2_000
MAX_FID_ELEMENTS = 40_000_000
MAX_FID_STATS_ARCHIVE_BYTES = 48 * 1024 * 1024
MAX_FID_STATS_MEMBER_BYTES = 34 * 1024 * 1024
MAX_FID_NPY_HEADER_BYTES = 4096
MAX_FID_DIRECTORY_ENTRIES = 100_000
MAX_FID_IMAGES = 50_100
MAX_FID_IMAGE_BYTES = 16 * 1024 * 1024
MAX_FID_TOTAL_IMAGE_BYTES = 2 * 1024 * 1024 * 1024
MAX_FID_IMAGE_PIXELS = 4_194_304
MAX_FID_TOTAL_IMAGE_PIXELS = 250_000_000
MAX_FID_ACTIVATIONS_BYTES = 1024 * 1024 * 1024
FID_IMAGE_EXTENSIONS = frozenset(
    {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}
)
_FID_CONSTRUCTION_LOCK = threading.Lock()


def _windows_fid_drive_type(root: str) -> int:
    import ctypes
    from ctypes import wintypes

    get_drive_type = ctypes.WinDLL("kernel32", use_last_error=True).GetDriveTypeW
    get_drive_type.argtypes = (wintypes.LPCWSTR,)
    get_drive_type.restype = wintypes.UINT
    return int(get_drive_type(root))


def _checked_local_fid_path(path: str) -> Path:
    """Reject Windows device paths and linked ancestors before opening input."""
    raw_path = os.fspath(path)
    if not raw_path:
        raise ValueError("FID input path is empty")
    if os.name == "nt" and raw_path.replace("/", "\\").startswith("\\\\"):
        raise ValueError("FID UNC or device paths are not supported")

    absolute = Path(os.path.abspath(raw_path))
    if os.name == "nt" and str(absolute).startswith("\\\\"):
        raise ValueError("FID UNC or device paths are not supported")
    if os.name == "nt":
        drive_type = _windows_fid_drive_type(absolute.anchor)
        if drive_type == 4:
            raise ValueError("FID remote drives are not supported")
        if drive_type not in {2, 3, 5, 6}:
            raise ValueError("FID drive type is unknown or invalid")
    current = Path(absolute.anchor)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & reparse_flag:
            raise ValueError("FID input has a linked or reparse ancestor")
    return absolute


def _load_local_fid_weights(path: str) -> dict:
    """Preflight locally supplied safetensors before model construction."""
    weights_path = _checked_local_fid_path(path)
    from safetensors import safe_open

    metadata = weights_path.lstat()
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)
    ):
        raise ValueError("FID weights must be a local regular safetensors file")
    if not 8 <= metadata.st_size <= MAX_FID_WEIGHTS_BYTES:
        raise ValueError("FID weights exceed the permitted file-size budget")

    with safe_open(str(weights_path), framework="pt", device="cpu") as source:
        keys = list(source.keys())
        if not keys or len(keys) > MAX_FID_TENSORS:
            raise ValueError("FID weights exceed the tensor-count budget")
        total_elements = 0
        for key in keys:
            tensor = source.get_slice(key)
            shape = tensor.get_shape()
            dtype = tensor.get_dtype()
            elements = math.prod(shape)
            if (
                len(shape) > 4
                or dtype not in {"F32", "I64"}
                or (dtype == "I64" and shape)
                or elements > MAX_FID_ELEMENTS
            ):
                raise ValueError("FID weights contain an unsupported tensor")
            total_elements += elements
            if total_elements > MAX_FID_ELEMENTS:
                raise ValueError("FID weights exceed the element budget")
        return {key: source.get_tensor(key) for key in keys}


def _is_local_fid_file(metadata: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return (
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and not bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)
    )


def _preflight_fid_stats(path: Path, dims: int, size: int) -> None:
    """Bound ZIP metadata and NPY shapes before pytorch-fid calls np.load."""
    import numpy as np

    if not 22 <= size <= MAX_FID_STATS_ARCHIVE_BYTES:
        raise ValueError("FID stats archive exceeds the file-size budget")

    # Read the small ZIP end record first: ZipFile would otherwise materialize
    # an attacker-controlled number of central-directory entries in memory.
    tail_start = max(0, size - 65_557)
    with path.open("rb") as source:
        source.seek(tail_start)
        tail = source.read()
    end_offset = tail.rfind(b"PK\x05\x06")
    if end_offset < 0 or end_offset + 22 > len(tail):
        raise ValueError("FID stats must be a small ZIP archive")
    _, disk, directory_disk, disk_entries, entries, directory_size, directory_offset, comment_size = (
        struct.unpack_from("<IHHHHIIH", tail, end_offset)
    )
    if (
        disk == 0xFFFF
        or directory_disk == 0xFFFF
        or disk_entries == 0xFFFF
        or entries == 0xFFFF
        or directory_size == 0xFFFFFFFF
        or directory_offset == 0xFFFFFFFF
    ):
        raise ValueError("FID stats ZIP64 archives are not supported")
    end_absolute = tail_start + end_offset
    with path.open("rb") as source:
        if end_absolute >= 20:
            source.seek(end_absolute - 20)
            if source.read(4) == b"PK\x06\x07":
                raise ValueError("FID stats ZIP64 locator is not supported")
        if end_absolute >= 76:
            source.seek(end_absolute - 76)
            if source.read(4) == b"PK\x06\x06":
                raise ValueError("FID stats ZIP64 end record is not supported")
    if (
        disk != 0
        or directory_disk != 0
        or disk_entries != 2
        or entries != 2
        or directory_size > 4096
        or end_offset + 22 + comment_size != len(tail)
        or directory_offset + directory_size != end_absolute
    ):
        raise ValueError("FID stats must contain exactly two bounded arrays")

    expected_shapes = {"mu.npy": (dims,), "sigma.npy": (dims, dims)}
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) != 2 or {item.filename for item in members} != set(expected_shapes):
                raise ValueError("FID stats must contain only mu and sigma")
            for member in members:
                if (
                    member.flag_bits & 1
                    or member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    or member.file_size > MAX_FID_STATS_MEMBER_BYTES
                ):
                    raise ValueError("FID stats contain an unsupported array")
                with archive.open(member) as array_source:
                    npy_version = np.lib.format.read_magic(array_source)
                    if npy_version == (1, 0):
                        length_format = "<H"
                        read_header = np.lib.format.read_array_header_1_0
                    elif npy_version == (2, 0):
                        length_format = "<I"
                        read_header = np.lib.format.read_array_header_2_0
                    else:
                        raise ValueError("FID stats use an unsupported NPY header")
                    length_size = struct.calcsize(length_format)
                    length_bytes = array_source.read(length_size)
                    if len(length_bytes) != length_size:
                        raise ValueError("FID stats NPY header is truncated")
                    header_size = struct.unpack(length_format, length_bytes)[0]
                    if header_size > MAX_FID_NPY_HEADER_BYTES:
                        raise ValueError("FID stats NPY header exceeds the size budget")
                    header = array_source.read(header_size)
                    if len(header) != header_size:
                        raise ValueError("FID stats NPY header is truncated")
                    shape, _, dtype = read_header(
                        io.BytesIO(length_bytes + header),
                        max_header_size=MAX_FID_NPY_HEADER_BYTES,
                    )
                    if shape != expected_shapes[member.filename]:
                        raise ValueError("FID stats array shape does not match dims")
                    if dtype.kind != "f" or dtype.itemsize not in {4, 8}:
                        raise ValueError("FID stats require float32 or float64 arrays")
                    array_size = 8 + length_size + header_size + math.prod(shape) * dtype.itemsize
                    if member.file_size != array_size:
                        raise ValueError("FID stats array byte size does not match its header")
    except (OSError, zipfile.BadZipFile, EOFError) as exc:
        raise ValueError("FID stats archive is invalid") from exc


def _preflight_fid_images(path: Path, dims: int) -> None:
    """Bound the image set before pytorch-fid opens files with Pillow."""
    from PIL import Image

    image_count = 0
    total_bytes = 0
    total_pixels = 0
    with os.scandir(path) as entries:
        for entry_count, entry in enumerate(entries, start=1):
            if entry_count > MAX_FID_DIRECTORY_ENTRIES:
                raise ValueError("FID image directory exceeds the entry-count budget")
            if entry.name.rsplit(".", 1)[-1].lower() not in FID_IMAGE_EXTENSIONS:
                continue
            metadata = entry.stat(follow_symlinks=False)
            if not _is_local_fid_file(metadata):
                raise ValueError("FID images must be local regular files")
            image_count += 1
            total_bytes += metadata.st_size
            if (
                image_count > MAX_FID_IMAGES
                or image_count * dims * 8 > MAX_FID_ACTIVATIONS_BYTES
                or not 0 < metadata.st_size <= MAX_FID_IMAGE_BYTES
                or total_bytes > MAX_FID_TOTAL_IMAGE_BYTES
            ):
                raise ValueError("FID images exceed the count or file-size budget")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(entry.path) as image:
                    width, height = image.size
                    pixels = width * height
                    total_pixels += pixels
                    if (
                        width < 1
                        or height < 1
                        or pixels > MAX_FID_IMAGE_PIXELS
                        or total_pixels > MAX_FID_TOTAL_IMAGE_PIXELS
                    ):
                        raise ValueError("FID image pixels exceed the budget")
                    image.verify()
    if not image_count:
        raise ValueError("FID image directory contains no supported images")


def _preflight_fid_path(path: str, dims: int) -> None:
    candidate = _checked_local_fid_path(path)
    metadata = candidate.lstat()
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & reparse_flag:
        raise ValueError("FID input must not be a symlink or reparse point")
    if candidate.suffix == ".npz":
        if not _is_local_fid_file(metadata):
            raise ValueError("FID stats must be a local regular .npz file")
        _preflight_fid_stats(candidate, dims, metadata.st_size)
    elif stat.S_ISDIR(metadata.st_mode):
        _preflight_fid_images(candidate, dims)
    else:
        raise ValueError("FID input must be an .npz stats file or image directory")


def load_model(
    checkpoint_path: str,
    device: torch.device,
    use_ema: bool = True,
) -> tuple:
    """Load a trained DDPM model from checkpoint.

    §4 — "we also report results with an exponential moving average"
    The EMA parameters typically produce better samples.

    Args:
        checkpoint_path: Path to the .safetensors file; strict JSON shares its stem
        device: Target device
        use_ema: Whether to load EMA parameters (recommended)

    Returns:
        (model, config_dict)
    """
    model, metadata = load_checkpoint(
        checkpoint_path,
        device=device,
        use_ema=use_ema,
    )
    if use_ema and metadata["has_ema"]:
        logger.info("Loaded EMA parameters")
    else:
        logger.info("Loaded model parameters (no EMA)")
    return model, metadata["config"]


@torch.no_grad()
def generate_samples(
    model: UNet,
    config: dict,
    num_samples: int = 64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate samples using Algorithm 2 — Sampling.

    §3.4 — "sampling from p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ²_t I)"

    Args:
        model: Trained UNet model
        config: Config dict from checkpoint
        num_samples: Number of images to generate
        device: Target device

    Returns:
        (num_samples, C, H, W) — generated images in [0, 1] range
    """
    if device is None:
        device = next(model.parameters()).device

    diff_cfg = config["diffusion"]
    data_cfg = config["data"]

    schedule = linear_noise_schedule(
        diff_cfg["timesteps"],
        diff_cfg["beta_start"],
        diff_cfg["beta_end"],
    )
    schedule = {name: tensor.to(device) for name, tensor in schedule.items()}

    image_size = data_cfg.get("image_size", 32)
    image_channels = config["model"].get("image_channels", 3)
    shape = (num_samples, image_channels, image_size, image_size)

    model.eval()
    samples = sample(model, schedule, shape, device)

    # Convert from [-1, 1] to [0, 1] for saving
    samples = (samples + 1.0) / 2.0
    samples = samples.clamp(0.0, 1.0)

    return samples


def save_samples(
    samples: torch.Tensor,
    output_dir: str,
    prefix: str = "sample",
    make_grid: bool = True,
    nrow: int = 8,
):
    """Save generated samples as images.

    Args:
        samples: (N, C, H, W) in [0, 1]
        output_dir: Directory to save images
        prefix: Filename prefix
        make_grid: If True, also save a grid image
        nrow: Number of images per row in grid
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if make_grid:
        grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2)
        grid_path = output_dir / f"{prefix}_grid.png"
        torchvision.utils.save_image(grid, grid_path)
        logger.info(f"Saved grid: {grid_path}")

    # Save individual images
    for i, img in enumerate(samples):
        img_path = output_dir / f"{prefix}_{i:05d}.png"
        torchvision.utils.save_image(img, img_path)

    logger.info(f"Saved {len(samples)} individual images to {output_dir}")


def compute_fid(
    generated_dir: str,
    real_stats_path: Optional[str] = None,
    batch_size: int = 50,
    device: str = "cuda",
    dims: int = 2048,
    inception_weights_path: Optional[str] = None,
) -> float:
    """Compute FID score between generated samples and real data.

    §4 — "We report FID score... Our best results are FID: 3.17"

    Requires pytorch-fid from the reviewed requirements-win-py313.lock and
    locally converted safetensors Inception weights. No model download occurs.

    For CIFAR-10, you need pre-computed stats for the real training set,
    or provide a directory of real images.

    Args:
        generated_dir: Directory containing generated .png images
        real_stats_path: Path to pre-computed .npz stats for real data,
                         OR directory containing real images
        batch_size: Batch size for Inception feature extraction
        device: Device for computation
        dims: Inception feature dimensionality (2048 = pool3)
        inception_weights_path: Local converted FID Inception safetensors file

    Returns:
        FID score (float). Lower is better.
    """
    if real_stats_path is None:
        raise ValueError(
            "Must provide real_stats_path: either a .npz file with pre-computed "
            "Inception statistics, or a directory of real CIFAR-10 images."
        )
    for input_path in (generated_dir, real_stats_path):
        if not _checked_local_fid_path(input_path).exists():
            raise RuntimeError(f"Invalid FID path: {input_path}")

    if inception_weights_path is None or not inception_weights_path.endswith(".safetensors"):
        raise ValueError("FID requires local Inception weights in safetensors format")
    _checked_local_fid_path(inception_weights_path)
    if version("pytorch-fid") != "0.3.0":
        raise RuntimeError("FID requires the reviewed pytorch-fid 0.3.0 release")

    _preflight_fid_path(generated_dir, dims)
    _preflight_fid_path(real_stats_path, dims)

    from pytorch_fid import fid_score, inception

    if inception.FID_WEIGHTS_URL != FID_WEIGHTS_URL:
        raise RuntimeError("the pytorch-fid Inception weight source changed")
    weights = _load_local_fid_weights(inception_weights_path)
    device_object = torch.device(device)
    block_index = inception.InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    # pytorch-fid 0.3.0 constructs Inception through this loader. Supply only
    # preflighted local tensors while its constructor runs; never call its
    # calculate_fid_given_paths helper, which would trigger a remote .pth load.
    with _FID_CONSTRUCTION_LOCK:
        original_loader = inception.load_state_dict_from_url

        def use_local_weights(url: str, *args, **kwargs):
            del args, kwargs
            if url != FID_WEIGHTS_URL:
                raise RuntimeError("unexpected Inception weight source")
            return weights

        try:
            inception.load_state_dict_from_url = use_local_weights
            model = inception.InceptionV3([block_index]).to(device_object)
        finally:
            inception.load_state_dict_from_url = original_loader
    del weights

    generated_mean, generated_covariance = fid_score.compute_statistics_of_path(
        generated_dir, model, batch_size, dims, device_object
    )
    real_mean, real_covariance = fid_score.compute_statistics_of_path(
        real_stats_path, model, batch_size, dims, device_object
    )
    fid = fid_score.calculate_frechet_distance(
        generated_mean,
        generated_covariance,
        real_mean,
        real_covariance,
    )

    logger.info(f"FID score: {fid:.2f}")
    return fid


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DDPM Evaluation — Generate samples and compute FID")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--num_samples", type=int, default=64,
                        help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="generated",
                        help="Output directory for generated images")
    parser.add_argument("--fid", action="store_true",
                        help="Compute FID score (requires --real_stats)")
    parser.add_argument("--real_stats", type=str, default=None,
                        help="Path to real data stats (.npz) or directory")
    parser.add_argument("--fid_weights", type=str, default=None,
                        help="Local converted FID Inception .safetensors weights")
    parser.add_argument("--no_ema", action="store_true",
                        help="Don't use EMA parameters")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, cfg = load_model(args.checkpoint, device, use_ema=not args.no_ema)
    samples = generate_samples(model, cfg, args.num_samples, device)
    save_samples(samples, args.output_dir)

    if args.fid:
        compute_fid(
            args.output_dir,
            args.real_stats,
            device=args.device,
            inception_weights_path=args.fid_weights,
        )

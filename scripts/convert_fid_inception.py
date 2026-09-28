#!/usr/bin/env python3
"""Explicitly convert the pinned upstream FID Inception weights to safetensors.

This offline preparation step is separate from evaluation. It accepts only the
known upstream .pth bytes, uses PyTorch's restricted weights loader, and never
downloads data or replaces an existing output file.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import stat
import tempfile
from pathlib import Path


SOURCE_BYTES = 95_628_359
SOURCE_SHA256 = "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"
MAX_TENSORS = 2_000
MAX_ELEMENTS = 40_000_000


def convert(source: Path, destination: Path) -> None:
    metadata = source.lstat()
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)
        or metadata.st_size != SOURCE_BYTES
    ):
        raise ValueError("source must be the pinned regular FID Inception weights file")
    if destination.suffix != ".safetensors" or os.path.lexists(destination):
        raise ValueError("destination must be a new .safetensors file")

    # Retain the verified bytes in memory so a path swap cannot change what
    # PyTorch deserializes after the digest check.
    with source.open("rb") as handle:
        body = handle.read(SOURCE_BYTES + 1)
    if len(body) != SOURCE_BYTES or hashlib.sha256(body).hexdigest() != SOURCE_SHA256:
        raise ValueError("source FID Inception weights failed SHA-256 verification")

    import torch
    from safetensors.torch import save_file

    state = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not 0 < len(state) <= MAX_TENSORS:
        raise ValueError("source weights have an invalid state dictionary")
    tensors = {}
    total_elements = 0
    for key, tensor in state.items():
        if (
            not isinstance(key, str)
            or not isinstance(tensor, torch.Tensor)
            or tensor.layout != torch.strided
            or tensor.dtype not in {torch.float32, torch.int64}
            or (tensor.dtype == torch.int64 and tensor.ndim != 0)
            or tensor.ndim > 4
        ):
            raise ValueError("source weights contain an unsupported tensor")
        total_elements += tensor.numel()
        if total_elements > MAX_ELEMENTS:
            raise ValueError("source weights exceed the element budget")
        tensors[key] = tensor.detach().contiguous()

    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        save_file(tensors, temp_name)
        os.link(temp_name, destination)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="manually acquired upstream .pth file")
    parser.add_argument("destination", type=Path, help="new local .safetensors file")
    args = parser.parse_args()
    try:
        convert(args.source, args.destination)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate paper2code's supported target manifests and hash locks."""

from __future__ import annotations

import re
import sys
from pathlib import Path


MAX_LOCK_BYTES = 1_000_000
HASH_PATTERN = re.compile(r"--hash=sha256:[0-9a-f]{64}(?=\s|$)")
PIN_PATTERN = re.compile(
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;\\]+)"
)
FORBIDDEN_REMOTE_MARKERS = (
    "://",
    "git+",
    "hg+",
    "svn+",
    "bzr+",
    " @ ",
)
FORBIDDEN_OPTIONS = (
    "-e",
    "--editable",
    "-r",
    "--requirement",
    "-c",
    "--constraint",
    "-i",
    "--index-url",
    "--extra-index-url",
    "--find-links",
    "--trusted-host",
)
LOCKS = {
    Path("requirements-runtime-win-py313.lock"): Path("requirements-runtime.in"),
    Path("requirements-ci-win-py313.lock"): Path("requirements-ci.in"),
    Path("skills/paper2code/worked/attention_is_all_you_need/requirements-win-py313.lock"):
        Path("skills/paper2code/worked/attention_is_all_you_need/requirements.in"),
    Path("skills/paper2code/worked/ddpm/requirements-win-py313.lock"):
        Path("skills/paper2code/worked/ddpm/requirements.in"),
}
REQUIRED_LOCK_PACKAGES = {
    Path("requirements-runtime-win-py313.lock"): {"requests"},
    Path("requirements-ci-win-py313.lock"): {"pytest", "requests"},
    Path("skills/paper2code/worked/attention_is_all_you_need/requirements-win-py313.lock"):
        {"pyyaml", "sacrebleu", "torch"},
    Path("skills/paper2code/worked/ddpm/requirements-win-py313.lock"):
        {"matplotlib", "pytorch-fid", "pyyaml", "safetensors", "torch", "torchvision"},
}


class LockValidationError(ValueError):
    """A dependency manifest is not safe for the documented install path."""


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _logical_entries(source: str) -> list[str]:
    entries: list[str] = []
    pending = ""
    for raw_line in source.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pending = f"{pending} {stripped}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        entries.append(pending)
        pending = ""
    if pending:
        raise LockValidationError("unterminated line continuation")
    return entries


def _reject_remote_or_option(entry: str, label: str) -> None:
    lowered = entry.casefold()
    if any(marker in lowered for marker in FORBIDDEN_REMOTE_MARKERS):
        raise LockValidationError(f"{label}: remote/direct dependency reference")
    first = lowered.split(maxsplit=1)[0]
    if first in FORBIDDEN_OPTIONS or any(
        first.startswith(f"{option}=") for option in FORBIDDEN_OPTIONS
    ):
        raise LockValidationError(f"{label}: dependency option/index is forbidden")
    if ";" in entry:
        raise LockValidationError(f"{label}: environment markers are forbidden in a target lock")


def verify_lock_text(source: str, *, label: str) -> dict[str, str]:
    """Validate strict exact pins and SHA-256 hashes in one lock's text."""
    if len(source.encode("utf-8")) > MAX_LOCK_BYTES:
        raise LockValidationError(f"{label}: lock exceeds its size budget")
    packages: dict[str, str] = {}
    entries = _logical_entries(source)
    if not entries:
        raise LockValidationError(f"{label}: lock is empty")
    for entry in entries:
        _reject_remote_or_option(entry, label)
        match = PIN_PATTERN.match(entry)
        if match is None or match.end() == len(entry):
            raise LockValidationError(f"{label}: every entry needs an exact pin and hash")
        suffix = entry[match.end():]
        hashes = HASH_PATTERN.findall(suffix)
        if not hashes or re.search(r"--hash=(?!sha256:)", suffix):
            raise LockValidationError(f"{label}: every entry needs only SHA-256 hashes")
        residue = HASH_PATTERN.sub("", suffix).strip()
        if residue:
            raise LockValidationError(f"{label}: unrecognized lock syntax")
        name = _canonical_name(match.group("name"))
        if name in packages:
            raise LockValidationError(f"{label}: duplicate package pin")
        packages[name] = match.group("version")
    return packages


def verify_input_text(source: str, *, label: str) -> dict[str, str]:
    """Validate a reviewed direct-input manifest without calling it a lock."""
    packages: dict[str, str] = {}
    entries = _logical_entries(source)
    if not entries:
        raise LockValidationError(f"{label}: input manifest is empty")
    for entry in entries:
        _reject_remote_or_option(entry, label)
        match = PIN_PATTERN.fullmatch(entry)
        if match is None:
            raise LockValidationError(f"{label}: direct inputs must use exact pins")
        name = _canonical_name(match.group("name"))
        if name in packages:
            raise LockValidationError(f"{label}: duplicate direct input")
        packages[name] = match.group("version")
    return packages


def verify_repository(root: Path) -> dict[Path, int]:
    """Validate all locks and ensure they contain their exact direct inputs."""
    counts: dict[Path, int] = {}
    for lock_relative, input_relative in LOCKS.items():
        lock_path = root / lock_relative
        input_path = root / input_relative
        if not lock_path.is_file() or not input_path.is_file():
            raise LockValidationError(f"missing supported manifest: {lock_relative}")
        locked = verify_lock_text(lock_path.read_text(encoding="utf-8"), label=str(lock_relative))
        direct = verify_input_text(input_path.read_text(encoding="utf-8"), label=str(input_relative))
        for name, version in direct.items():
            if locked.get(name) != version:
                raise LockValidationError(f"{lock_relative}: direct input is absent or changed")
        missing = REQUIRED_LOCK_PACKAGES[lock_relative] - set(locked)
        if missing:
            raise LockValidationError(f"{lock_relative}: required runtime package is absent")
        counts[lock_relative] = len(locked)
    return counts


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        counts = verify_repository(root)
    except (OSError, UnicodeError, LockValidationError) as exc:
        print(f"dependency lock validation failed: {exc}", file=sys.stderr)
        return 1
    for path, count in counts.items():
        print(f"validated {path.as_posix()}: {count} exact hashed packages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

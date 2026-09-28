from __future__ import annotations

from pathlib import Path

import pytest

from scripts.verify_dependency_locks import LockValidationError, verify_lock_text, verify_repository


ROOT = Path(__file__).parents[1]


def test_repository_dependency_locks_are_strict_and_complete() -> None:
    verify_repository(ROOT)


@pytest.mark.parametrize(
    "source",
    [
        "example>=1.0 --hash=sha256:" + "a" * 64,
        "example @ https://example.invalid/example.whl --hash=sha256:" + "a" * 64,
        "git+https://example.invalid/repository.git",
        "--extra-index-url https://example.invalid/simple\nexample==1.0 --hash=sha256:" + "a" * 64,
        "example==1.0",
        "example==1.0 --hash=md5:" + "a" * 32,
    ],
)
def test_lock_validator_rejects_floating_remote_or_unhashed_entries(source: str) -> None:
    with pytest.raises(LockValidationError):
        verify_lock_text(source, label="fixture.lock")

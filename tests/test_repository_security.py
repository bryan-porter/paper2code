from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_install_instructions_use_hardened_fork_and_warn_about_floating_source() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "npx skills@1.5.24 add bryan-porter/paper2code/skills/paper2code" in readme
    assert "PrathamLearnsToCode/paper2code/skills/paper2code" not in readme
    assert "unpinned" in readme.lower()
    assert "review" in readme.lower()


def test_skill_does_not_recommend_unpinned_direct_pip_installs() -> None:
    unsafe: list[str] = []
    pattern = re.compile(
        r"(?<![\w-])pip(?:\s+--[\w-]+)*\s+install\b[^\r\n]*"
    )
    for path in (ROOT / "skills" / "paper2code").rglob("*"):
        if not path.is_file() or path.suffix not in {".md", ".py"}:
            continue
        source = path.read_text(encoding="utf-8")
        for match in pattern.finditer(source):
            command = match.group(0)
            if all(
                required in command
                for required in (
                    "pip --isolated install",
                    "--require-hashes",
                    "--only-binary=:all:",
                    "--index-url https://pypi.org/simple",
                    "-r ",
                )
            ):
                continue
            line = source.count("\n", 0, match.start()) + 1
            unsafe.append(f"{path.relative_to(ROOT)}:{line}")

    assert unsafe == [], "unpinned direct pip install recommendation(s): " + ", ".join(unsafe)


def test_ddpm_checkpoints_never_use_pickle_serialization() -> None:
    unsafe: list[str] = []
    ddpm_root = ROOT / "skills" / "paper2code" / "worked" / "ddpm"
    for path in ddpm_root.rglob("*.py"):
        if any(
                part in {".git", ".paper2code_work", ".venv", ".lock-venv", ".verify-venvs"}
            for part in path.parts
        ):
            continue
        source = path.read_text(encoding="utf-8")
        if "torch.load(" not in source and "torch.save(" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
                and node.func.attr in {"load", "save"}
            ):
                unsafe.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert unsafe == [], "pickle checkpoint call(s): " + ", ".join(unsafe)


def test_skill_never_auto_installs_remote_packages() -> None:
    skill = (ROOT / "skills" / "paper2code" / "SKILL.md").read_text(encoding="utf-8")

    assert "pip install pymupdf4llm" not in skill
    assert "Do not install packages automatically" in skill
    assert "untrusted external content" in skill.lower()


def test_skill_never_opens_downloaded_pdf_with_a_parser() -> None:
    offenders: list[str] = []
    parser_names = ("pymupdf" + "4llm", "pdf" + "plumber")
    for path in (ROOT / "skills" / "paper2code").rglob("*"):
        if path.is_file() and path.suffix in {".md", ".py", ".ipynb", ".txt"}:
            lowered = path.read_text(encoding="utf-8").lower()
            if any(name in lowered for name in parser_names):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], "downloaded-PDF parser reference(s): " + ", ".join(offenders)


def test_repository_does_not_direct_users_to_plaintext_http() -> None:
    offenders: list[str] = []
    tracked_and_untracked = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8").split("\0")
    for relative_name in tracked_and_untracked:
        if not relative_name:
            continue
        path = ROOT / relative_name
        relative_path = Path(relative_name)
        if (
            not path.is_file()
            or relative_path.parts[0] == "tests"
            or path.suffix not in {".py", ".md"}
        ):
            continue
        if ("http" + "://") in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], "plaintext HTTP reference(s): " + ", ".join(offenders)


def test_supported_runtime_has_complete_hash_locked_closures() -> None:
    lock_paths = [
        ROOT / "requirements-runtime-win-py313.lock",
        ROOT / "requirements-ci-win-py313.lock",
        ROOT / "skills" / "paper2code" / "worked" / "attention_is_all_you_need" / "requirements-win-py313.lock",
        ROOT / "skills" / "paper2code" / "worked" / "ddpm" / "requirements-win-py313.lock",
    ]
    for path in lock_paths:
        assert path.is_file(), path
        source = path.read_text(encoding="utf-8")
        assert "--hash=sha256:" in source, path
        assert "git+" not in source and " @ " not in source, path
        assert "--extra-index-url" not in source and "--index-url" not in source, path

    ddpm = lock_paths[-1].read_text(encoding="utf-8")
    assert "safetensors==" in ddpm
    assert "torchvision==" in ddpm


def test_github_actions_are_immutable_and_least_privilege() -> None:
    workflows = sorted((ROOT / ".github" / "workflows").glob("*.y*ml"))
    assert workflows

    for path in workflows:
        workflow = path.read_text(encoding="utf-8")
        assert re.search(r"(?m)^permissions:\s*\n\s{2}contents: read\s*$", workflow)
        action_refs = re.findall(r"(?m)^\s*-?\s*uses:\s*[^@\s]+@([^\s#]+)", workflow)
        assert action_refs, path
        assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in action_refs), path
        assert "persist-credentials: false" in workflow
        assert re.search(r"(?m)^\s{4}timeout-minutes: [1-9]\d*\s*$", workflow)


def test_security_sensitive_paths_have_explicit_owners() -> None:
    codeowners = (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8")

    for protected_path in (
        "/.github/",
        "/requirements-*.in",
        "/requirements-*.lock",
        "/skills/paper2code/worked/*/requirements.in",
        "/skills/paper2code/worked/*/requirements-*.lock",
        "/skills/paper2code/SKILL.md",
        "/skills/paper2code/scripts/",
    ):
        assert re.search(
            rf"(?m)^{re.escape(protected_path)}\s+@bryan-porter\s*$",
            codeowners,
        )


def test_worked_notebooks_have_no_saved_execution_output() -> None:
    notebooks = sorted((ROOT / "skills" / "paper2code" / "worked").glob("*/notebooks/*.ipynb"))
    assert notebooks

    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            assert cell.get("execution_count") is None, path
            assert cell.get("outputs") == [], path

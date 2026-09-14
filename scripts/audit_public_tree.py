"""Fail when publication-excluded files or private path patterns enter Git."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".ckpt", ".pt", ".pth", ".tif", ".tiff", ".zip"}
FORBIDDEN_TEXT = (
    "Aetosky" + "_colab",
    "x-access" + "-token",
    "github_" + "pat_",
    "ghp" + "_",
)
TEXT_SUFFIXES = {
    ".cff",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def main() -> int:
    failures: list[str] = []
    for path in tracked_files():
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"publication-excluded binary is tracked: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in FORBIDDEN_TEXT:
            if pattern in text:
                failures.append(f"private/credential pattern in {relative}: {pattern}")
    if failures:
        raise SystemExit("\n".join(failures))
    print("Public-tree audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def stable_hash(payload: Any) -> str:
    """Return a deterministic SHA-256 for JSON-compatible content."""
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, target)


def verify_files(
    root: str | Path,
    file_specs: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Verify size/checksum for files declared by a dataset contract."""
    root = Path(root)
    results: list[dict[str, Any]] = []
    failures: list[str] = []

    for spec in file_specs:
        role = str(spec["role"])
        relative_path = Path(str(spec["path"]))
        path = root / relative_path
        expected_sha = str(spec.get("sha256", "")).lower()

        row: dict[str, Any] = {
            "role": role,
            "path": str(path),
            "exists": path.is_file(),
        }
        if not path.is_file():
            row["status"] = "missing"
            failures.append(f"{role}: missing {path}")
            results.append(row)
            continue

        row["bytes"] = path.stat().st_size
        actual_sha = sha256_file(path)
        row["sha256"] = actual_sha
        if expected_sha and actual_sha != expected_sha:
            row["status"] = "sha256_mismatch"
            failures.append(
                f"{role}: expected {expected_sha}, received {actual_sha}"
            )
        else:
            row["status"] = "ok"
        results.append(row)

    if failures:
        raise ValueError("Dataset contract violation:\n- " + "\n- ".join(failures))
    return results

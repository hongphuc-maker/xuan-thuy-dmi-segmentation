import hashlib
from pathlib import Path

import pytest

from xuanthuy_seg.contracts import stable_hash, verify_files


def test_stable_hash_ignores_mapping_order() -> None:
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_verify_files_detects_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"correct")
    expected = hashlib.sha256(b"correct").hexdigest()
    result = verify_files(tmp_path, [{"role": "image", "path": "input.bin", "sha256": expected}])
    assert result[0]["status"] == "ok"
    with pytest.raises(ValueError, match="Dataset contract violation"):
        verify_files(tmp_path, [{"role": "image", "path": "input.bin", "sha256": "0" * 64}])

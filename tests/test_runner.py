import pytest

pytest.importorskip("torch")
pytest.importorskip("pandas")
pytest.importorskip("rasterio")

from xuanthuy_seg.training.runner import _build_schedule


def test_global_schedule_reuses_each_frozen_batch_once_per_pass() -> None:
    schedule = _build_schedule(maximum_steps=10, n_batches=4, seed=123)
    assert [row["batch_index"] for row in schedule[:4]] == [0, 1, 2, 3]
    assert sorted(row["batch_index"] for row in schedule[4:8]) == [0, 1, 2, 3]
    assert schedule == _build_schedule(maximum_steps=10, n_batches=4, seed=123)

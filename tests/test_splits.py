import numpy as np

from xuanthuy_seg.data.splits import (
    GUARD,
    TRAIN,
    VALIDATION,
    build_split_domains,
    build_split_mask,
)


def test_vertical_band_split_has_guard_and_disjoint_domains() -> None:
    valid = np.ones((8, 20), dtype=bool)
    config = {
        "strategy": "vertical_band",
        "parameters": {
            "validation_start": 8,
            "validation_stop": 12,
            "guard_px_each_side": 2,
        },
    }
    mask = build_split_mask(valid.shape, valid, config)
    assert np.all(mask[:, 8:12] == VALIDATION)
    assert np.all(mask[:, 6:8] == GUARD)
    assert np.all(mask[:, 12:14] == GUARD)
    assert np.all(mask[:, :6] == TRAIN)
    assert np.all(mask[:, 14:] == TRAIN)


def test_spatial_block_kfold_is_reproducible() -> None:
    valid = np.ones((24, 24), dtype=bool)
    config = {
        "strategy": "spatial_block_kfold",
        "parameters": {
            "block_height": 6,
            "block_width": 6,
            "n_folds": 4,
            "validation_fold": 1,
            "seed": 17,
            "guard_px": 1,
        },
    }
    first = build_split_mask(valid.shape, valid, config)
    second = build_split_mask(valid.shape, valid, config)
    assert np.array_equal(first, second)
    assert np.any(first == TRAIN)
    assert np.any(first == VALIDATION)
    assert np.any(first == GUARD)


def test_generated_band_input_domains_share_no_pixels() -> None:
    valid = np.ones((32, 80), dtype=bool)
    config = {
        "strategy": "vertical_band",
        "parameters": {
            "validation_start": 30,
            "validation_stop": 50,
            "guard_px_each_side": 5,
        },
    }
    domains = build_split_domains(valid.shape, valid, config)
    assert not np.any(domains["train_input"] & domains["validation_input"])
    assert np.all(domains["validation_metric"][:, 30:50])
    assert np.all(domains["guard"][:, 25:30])
    assert np.all(domains["guard"][:, 50:55])


def test_generated_block_domains_are_deterministic_and_disjoint() -> None:
    valid = np.ones((24, 24), dtype=bool)
    config = {
        "strategy": "spatial_block_kfold",
        "parameters": {
            "block_height": 6,
            "block_width": 6,
            "n_folds": 4,
            "validation_fold": 1,
            "seed": 17,
            "guard_px": 1,
        },
    }
    first = build_split_domains(valid.shape, valid, config)
    second = build_split_domains(valid.shape, valid, config)
    for key in first:
        assert np.array_equal(first[key], second[key])
    assert not np.any(first["train_input"] & first["validation_input"])

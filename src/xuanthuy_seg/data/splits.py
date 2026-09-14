from __future__ import annotations

from typing import Any

import numpy as np

EXCLUDED = np.uint8(0)
TRAIN = np.uint8(1)
VALIDATION = np.uint8(2)
GUARD = np.uint8(3)


def _band_split(
    shape: tuple[int, int],
    valid_mask: np.ndarray,
    parameters: dict[str, Any],
    axis: int,
) -> np.ndarray:
    height, width = shape
    extent = width if axis == 1 else height
    start = int(parameters["validation_start"])
    stop = int(parameters["validation_stop"])
    guard = int(parameters["guard_px_each_side"])
    if not 0 <= start < stop <= extent:
        raise ValueError(f"Invalid validation interval [{start}, {stop}) for extent {extent}")
    if guard < 0:
        raise ValueError("guard_px_each_side cannot be negative")

    coordinates = np.arange(extent)
    validation_1d = (coordinates >= start) & (coordinates < stop)
    guard_1d = (
        ((coordinates >= max(0, start - guard)) & (coordinates < start))
        | ((coordinates >= stop) & (coordinates < min(extent, stop + guard)))
    )
    if axis == 1:
        validation = np.broadcast_to(validation_1d[None, :], shape)
        guard_mask = np.broadcast_to(guard_1d[None, :], shape)
    else:
        validation = np.broadcast_to(validation_1d[:, None], shape)
        guard_mask = np.broadcast_to(guard_1d[:, None], shape)

    output = np.full(shape, EXCLUDED, dtype=np.uint8)
    output[valid_mask] = TRAIN
    output[valid_mask & guard_mask] = GUARD
    output[valid_mask & validation] = VALIDATION
    return output


def _spatial_block_kfold(
    shape: tuple[int, int],
    valid_mask: np.ndarray,
    parameters: dict[str, Any],
) -> np.ndarray:
    try:
        from scipy.ndimage import distance_transform_edt
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("spatial_block_kfold requires scipy") from error

    height, width = shape
    block_height = int(parameters["block_height"])
    block_width = int(parameters["block_width"])
    n_folds = int(parameters["n_folds"])
    validation_fold = int(parameters["validation_fold"])
    seed = int(parameters["seed"])
    guard_px = int(parameters["guard_px"])
    if min(block_height, block_width, n_folds) <= 0:
        raise ValueError("block dimensions and n_folds must be positive")
    if not 0 <= validation_fold < n_folds:
        raise ValueError("validation_fold must be in [0, n_folds)")

    rows = np.arange(height)[:, None] // block_height
    cols = np.arange(width)[None, :] // block_width
    n_block_cols = int(np.ceil(width / block_width))
    block_ids = rows * n_block_cols + cols
    unique_blocks = np.unique(block_ids[valid_mask])
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_blocks)
    fold_by_block = {int(block): index % n_folds for index, block in enumerate(shuffled)}
    validation_blocks = np.array(
        [block for block, fold in fold_by_block.items() if fold == validation_fold],
        dtype=block_ids.dtype,
    )
    validation = valid_mask & np.isin(block_ids, validation_blocks)
    if guard_px:
        expanded = distance_transform_edt(~validation) <= guard_px
        guard = valid_mask & expanded & ~validation
    else:
        guard = np.zeros(shape, dtype=bool)

    output = np.full(shape, EXCLUDED, dtype=np.uint8)
    output[valid_mask] = TRAIN
    output[guard] = GUARD
    output[validation] = VALIDATION
    return output


def build_split_mask(
    shape: tuple[int, int],
    valid_mask: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Build a pixel-domain spatial split from a versioned strategy config."""
    if valid_mask.shape != shape or valid_mask.dtype != bool:
        raise ValueError("valid_mask must be bool and have the requested raster shape")
    strategy = str(config["strategy"])
    parameters = config["parameters"]
    if strategy == "vertical_band":
        output = _band_split(shape, valid_mask, parameters, axis=1)
    elif strategy == "horizontal_band":
        output = _band_split(shape, valid_mask, parameters, axis=0)
    elif strategy == "spatial_block_kfold":
        output = _spatial_block_kfold(shape, valid_mask, parameters)
    elif strategy == "external_mask":
        raise ValueError("external_mask must be loaded and verified by the raster IO layer")
    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")

    if np.any((output == TRAIN) & (output == VALIDATION)):
        raise AssertionError("Train and validation overlap")
    if not np.any(output == TRAIN) or not np.any(output == VALIDATION):
        raise ValueError("A split must contain both train and validation pixels")
    return output


def build_split_domains(
    shape: tuple[int, int],
    valid_mask: np.ndarray,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Return target and input domains used by the artifact generator.

    ``train_input`` and ``validation_input`` are geometry masks independent of
    nodata. Requiring every crop to remain inside one of these domains makes
    the zero-shared-input-pixel rule directly auditable.
    """
    if valid_mask.shape != shape or valid_mask.dtype != bool:
        raise ValueError("valid_mask must be bool and have the requested raster shape")
    height, width = shape
    strategy = str(config["strategy"])
    parameters = config["parameters"]

    if strategy in {"vertical_band", "horizontal_band"}:
        axis = 1 if strategy == "vertical_band" else 0
        extent = width if axis == 1 else height
        start = int(parameters["validation_start"])
        stop = int(parameters["validation_stop"])
        guard = int(parameters["guard_px_each_side"])
        if not 0 <= start < stop <= extent:
            raise ValueError(f"Invalid validation interval [{start}, {stop})")
        coordinates = np.arange(extent)
        metric_1d = (coordinates >= start) & (coordinates < stop)
        input_1d = (
            (coordinates >= max(0, start - guard))
            & (coordinates < min(extent, stop + guard))
        )
        if axis == 1:
            validation_metric = np.broadcast_to(metric_1d[None, :], shape).copy()
            validation_input = np.broadcast_to(input_1d[None, :], shape).copy()
        else:
            validation_metric = np.broadcast_to(metric_1d[:, None], shape).copy()
            validation_input = np.broadcast_to(input_1d[:, None], shape).copy()
        guard_domain = validation_input & ~validation_metric
        train_input = ~validation_input
    elif strategy == "spatial_block_kfold":
        try:
            from scipy.ndimage import distance_transform_edt
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError("spatial_block_kfold requires scipy") from error
        block_height = int(parameters["block_height"])
        block_width = int(parameters["block_width"])
        n_folds = int(parameters["n_folds"])
        validation_fold = int(parameters["validation_fold"])
        seed = int(parameters["seed"])
        guard_px = int(parameters.get("guard_px", 0))
        rows = np.arange(height)[:, None] // block_height
        cols = np.arange(width)[None, :] // block_width
        n_block_cols = int(np.ceil(width / block_width))
        block_ids = rows * n_block_cols + cols
        unique_blocks = np.unique(block_ids[valid_mask])
        shuffled = np.random.default_rng(seed).permutation(unique_blocks)
        fold_by_block = {int(block): index % n_folds for index, block in enumerate(shuffled)}
        validation_blocks = np.asarray(
            [block for block, fold in fold_by_block.items() if fold == validation_fold],
            dtype=block_ids.dtype,
        )
        validation_input = np.isin(block_ids, validation_blocks)
        validation_metric = validation_input.copy()
        guard_domain = (
            (distance_transform_edt(~validation_input) <= guard_px) & ~validation_input
            if guard_px
            else np.zeros(shape, dtype=bool)
        )
        train_input = ~validation_input & ~guard_domain
    else:
        raise ValueError(f"Unsupported split strategy for generated artifacts: {strategy}")

    split_mask = np.full(shape, EXCLUDED, dtype=np.uint8)
    split_mask[valid_mask & train_input] = TRAIN
    split_mask[valid_mask & guard_domain] = GUARD
    split_mask[valid_mask & validation_metric] = VALIDATION
    if np.any(train_input & validation_input):
        raise AssertionError("Train and validation input domains overlap")
    return {
        "split_mask": split_mask,
        "train_input": train_input,
        "validation_input": validation_input,
        "validation_metric": validation_metric,
        "guard": guard_domain,
    }


def split_summary(mask: np.ndarray) -> dict[str, int]:
    return {
        "excluded": int(np.count_nonzero(mask == EXCLUDED)),
        "train": int(np.count_nonzero(mask == TRAIN)),
        "validation": int(np.count_nonzero(mask == VALIDATION)),
        "guard": int(np.count_nonzero(mask == GUARD)),
    }

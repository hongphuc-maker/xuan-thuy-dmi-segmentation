from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio

from ..config import ConfigBundle
from ..contracts import atomic_json, sha256_file, stable_hash, verify_files
from .splits import build_split_domains


SCHEMA_VERSION = "xtseg-prepared-data-v1"
IMPLEMENTATION_REVISION = "xtseg-prepare-20260908-r1"


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _atomic_raster(array: np.ndarray, path: Path, reference_profile: dict[str, Any]) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    profile = reference_profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=str(array.dtype),
        nodata=0,
        compress="deflate",
        predictor=1,
    )
    with rasterio.open(temporary, "w", **profile) as destination:
        destination.write(array, 1)
    os.replace(temporary, path)


def _role_spec(dataset: dict[str, Any], role: str) -> dict[str, Any]:
    matches = [item for item in dataset["files"] if str(item["role"]) == role]
    if len(matches) != 1:
        raise ValueError(f"Dataset must define exactly one {role!r} file")
    return matches[0]


def _integral(array: np.ndarray) -> np.ndarray:
    return np.pad(
        array.astype(np.int64).cumsum(axis=0).cumsum(axis=1),
        ((1, 0), (1, 0)),
    )


def _all_window_sums(array: np.ndarray, size: int) -> np.ndarray:
    integral = _integral(array)
    return (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )


def _query_window_sums(integral: np.ndarray, rows: np.ndarray, cols: np.ndarray, size: int) -> np.ndarray:
    return (
        integral[rows + size, cols + size]
        - integral[rows, cols + size]
        - integral[rows + size, cols]
        + integral[rows, cols]
    )


def _cover_starts(start: int, stop: int, size: int, stride: int) -> list[int]:
    extent = stop - start
    if extent <= size:
        return [start]
    starts = list(range(start, stop - size + 1, stride))
    final = stop - size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _validation_regions(
    shape: tuple[int, int],
    split: dict[str, Any],
    validation_input: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    height, width = shape
    strategy = str(split["strategy"])
    parameters = split["parameters"]
    if strategy == "vertical_band":
        guard = int(parameters["guard_px_each_side"])
        return [(0, height, max(0, int(parameters["validation_start"]) - guard),
                 min(width, int(parameters["validation_stop"]) + guard))]
    if strategy == "horizontal_band":
        guard = int(parameters["guard_px_each_side"])
        return [(max(0, int(parameters["validation_start"]) - guard),
                 min(height, int(parameters["validation_stop"]) + guard), 0, width)]
    if strategy == "spatial_block_kfold":
        block_height = int(parameters["block_height"])
        block_width = int(parameters["block_width"])
        regions: list[tuple[int, int, int, int]] = []
        for row0 in range(0, height, block_height):
            for col0 in range(0, width, block_width):
                row1, col1 = min(height, row0 + block_height), min(width, col0 + block_width)
                if validation_input[row0:row1, col0:col1].any():
                    regions.append((row0, row1, col0, col1))
        return regions
    raise ValueError(f"Unsupported split strategy: {strategy}")


def _build_validation_manifest(
    shape: tuple[int, int],
    split: dict[str, Any],
    image_valid: np.ndarray,
    metric_mask: np.ndarray,
    validation_input: np.ndarray,
) -> pd.DataFrame:
    size = int(split["patch_policy"]["size"])
    stride = int(split["patch_policy"].get("validation_stride", size // 2))
    coverage = np.zeros(shape, dtype=np.uint16)
    rows: list[dict[str, Any]] = []
    for region_index, (row_start, row_stop, col_start, col_stop) in enumerate(
        _validation_regions(shape, split, validation_input)
    ):
        for row0 in _cover_starts(row_start, row_stop, size, stride):
            for col0 in _cover_starts(col_start, col_stop, size, stride):
                height = min(size, row_stop - row0)
                width = min(size, col_stop - col0)
                region = np.s_[row0 : row0 + height, col0 : col0 + width]
                if not validation_input[region].all():
                    raise ValueError("A validation window crosses its validation input domain")
                valid_pixels = int(image_valid[region].sum())
                metric_pixels = int(metric_mask[region].sum())
                if valid_pixels == 0 and metric_pixels == 0:
                    continue
                coverage[region] += validation_input[region].astype(np.uint16)
                rows.append(
                    {
                        "window_id": f"val_g{region_index:03d}_r{row0:04d}_c{col0:04d}",
                        "region_index": region_index,
                        "row": row0,
                        "col": col0,
                        "height": height,
                        "width": width,
                        "image_valid_px": valid_pixels,
                        "image_valid_fraction": valid_pixels / max(height * width, 1),
                        "validation_metric_px": metric_pixels,
                    }
                )
    if not rows or not np.all(coverage[metric_mask] > 0):
        raise ValueError("Generated validation windows do not cover every validation metric pixel")
    return pd.DataFrame(rows)


def _sample_manifest(
    label: np.ndarray,
    common_valid: np.ndarray,
    split_mask: np.ndarray,
    train_input: np.ndarray,
    split: dict[str, Any],
    class_map: dict[int, str],
) -> pd.DataFrame:
    patch = int(split["patch_policy"]["size"])
    if patch % 2:
        raise ValueError("The current center convention requires an even patch size")
    half = patch // 2
    minimum_fraction = float(split["patch_policy"].get("minimum_train_fraction", 1.0))
    sampling = split["sampling"]
    n_patches = int(sampling["n_patches"])
    rare_codes = [int(code) for code in sampling.get("rare_codes", [])]
    rare_quota = int(sampling.get("rare_quota_each", 0))
    support_floor = int(sampling.get("dmi_support_floor_px", 1))
    rng = np.random.default_rng(int(sampling["manifest_seed"]))

    footprint_ok = _all_window_sums(train_input, patch) == patch * patch
    train_target = common_valid & (split_mask == 1)
    target_counts = _all_window_sums(train_target, patch)
    footprint_ok &= target_counts >= math.ceil(minimum_fraction * patch * patch)
    top_rows, left_cols = np.nonzero(footprint_ok)
    center_rows, center_cols = top_rows + half, left_cols + half
    center_ok = train_target[center_rows, center_cols]
    top_rows, left_cols = top_rows[center_ok], left_cols[center_ok]
    center_rows, center_cols = center_rows[center_ok], center_cols[center_ok]
    center_classes = label[center_rows, center_cols].astype(np.int16)
    if len(top_rows) < n_patches:
        raise ValueError(f"Only {len(top_rows)} eligible centers for {n_patches} requested patches")

    rare_support: dict[int, np.ndarray] = {}
    for code in rare_codes:
        rare_support[code] = _query_window_sums(
            _integral(train_target & (label == code)),
            top_rows,
            left_cols,
            patch,
        )

    selected: list[int] = []
    streams: dict[int, str] = {}
    used = np.zeros(len(top_rows), dtype=bool)
    for code in rare_codes:
        eligible = np.flatnonzero((rare_support[code] >= support_floor) & ~used)
        if len(eligible) < rare_quota:
            raise ValueError(
                f"Class {code} has only {len(eligible)} eligible support patches; "
                f"rare_quota_each={rare_quota}"
            )
        chosen = rng.choice(eligible, size=rare_quota, replace=False)
        used[chosen] = True
        for index in chosen.tolist():
            streams[index] = f"rare_{code}_{class_map[code]}"
        selected.extend(chosen.tolist())

    remainder = n_patches - len(selected)
    if remainder < 0:
        raise ValueError("Rare quotas exceed n_patches")
    pool = np.flatnonzero(~used)
    chosen_remainder = rng.choice(pool, size=remainder, replace=False)
    for index in chosen_remainder.tolist():
        streams[index] = "uniform_remainder"
    selected.extend(chosen_remainder.tolist())
    selected_array = np.asarray(selected, dtype=np.int64)

    chosen_rows = top_rows[selected_array]
    chosen_cols = left_cols[selected_array]
    records: dict[str, Any] = {
        "patch_id": [
            f"train_cr{row:04d}_cc{col:04d}"
            for row, col in zip(center_rows[selected_array], center_cols[selected_array], strict=True)
        ],
        "center_row": center_rows[selected_array],
        "center_col": center_cols[selected_array],
        "row": chosen_rows,
        "col": chosen_cols,
        "height": patch,
        "width": patch,
        "center_class": center_classes[selected_array],
        "sampling_stream": [streams[index] for index in selected],
    }
    manifest = pd.DataFrame(records)
    manifest["train_valid_px"] = _query_window_sums(
        _integral(train_target), chosen_rows, chosen_cols, patch
    )
    manifest["train_valid_fraction"] = manifest["train_valid_px"] / (patch * patch)
    class_columns: list[str] = []
    for code in sorted(class_map):
        column = f"class_{code}_px"
        class_columns.append(column)
        manifest[column] = _query_window_sums(
            _integral(train_target & (label == code)),
            chosen_rows,
            chosen_cols,
            patch,
        )
    manifest["n_classes"] = (manifest[class_columns].to_numpy() > 0).sum(axis=1)
    manifest["class_codes"] = [
        "|".join(str(code) for code, count in zip(sorted(class_map), values, strict=True) if count > 0)
        for values in manifest[class_columns].to_numpy()
    ]
    if manifest.patch_id.duplicated().any():
        raise AssertionError("Duplicate patch centers were selected")
    return manifest


def _build_schedule(manifest: pd.DataFrame, split: dict[str, Any], class_map: dict[int, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sampling = split["sampling"]
    batch_size = int(sampling["batch_size"])
    floor = int(sampling.get("dmi_support_floor_px", 1))
    rare_codes = [int(code) for code in sampling.get("rare_codes", [])]
    n_patches = len(manifest)
    if n_patches % batch_size:
        raise ValueError("n_patches must be divisible by batch_size")
    n_batches = n_patches // batch_size
    if batch_size <= len(rare_codes):
        raise ValueError("batch_size must exceed the number of rare anchors")

    anchor_by_code: dict[int, np.ndarray] = {}
    reserved: set[int] = set()
    base_rng = np.random.default_rng(int(sampling["schedule_seed"]))
    for code in rare_codes:
        candidates = manifest.index[
            manifest["sampling_stream"].str.startswith(f"rare_{code}_")
            & (manifest[f"class_{code}_px"] >= floor)
        ].to_numpy(np.int64)
        if len(candidates) < n_batches:
            raise ValueError(f"Not enough class-{code} anchors for {n_batches} batches")
        anchors = base_rng.permutation(candidates)[:n_batches]
        if any(int(index) in reserved for index in anchors):
            raise AssertionError("Rare anchor pools unexpectedly overlap")
        reserved.update(int(index) for index in anchors)
        anchor_by_code[code] = anchors

    extras = np.asarray([index for index in manifest.index if int(index) not in reserved], dtype=np.int64)
    slots_remaining = batch_size - len(rare_codes)
    class_columns = [f"class_{code}_px" for code in sorted(class_map)]
    accepted_batches: list[list[int]] | None = None
    accepted_attempt = -1
    for attempt in range(512):
        rng = np.random.default_rng(int(sampling["schedule_seed"]) + attempt)
        shuffled = rng.permutation(extras)
        batches: list[list[int]] = []
        cursor = 0
        for batch_index in range(n_batches):
            batch = [int(anchor_by_code[code][batch_index]) for code in rare_codes]
            batch.extend(int(value) for value in shuffled[cursor : cursor + slots_remaining])
            cursor += slots_remaining
            batches.append(batch)
        exposures = np.asarray(
            [manifest.loc[batch, class_columns].to_numpy(np.int64).sum(axis=0) for batch in batches]
        )
        if np.all(exposures >= floor):
            accepted_batches = batches
            accepted_attempt = attempt
            break
    if accepted_batches is None:
        raise ValueError(
            "Could not construct class-complete batches after 512 deterministic attempts; "
            "increase patch count/batch size or revise rare sampling"
        )

    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(accepted_batches):
        anchor_lookup = {
            int(anchor_by_code[code][batch_index]): code for code in rare_codes
        }
        for slot, manifest_index in enumerate(batch):
            rows.append(
                {
                    "pass_index": 0,
                    "batch_index": batch_index,
                    "slot": slot,
                    "manifest_index": manifest_index,
                    "patch_id": manifest.loc[manifest_index, "patch_id"],
                    "anchor_code": anchor_lookup.get(manifest_index, 0),
                    "schedule_attempt": accepted_attempt,
                }
            )
        exposure = manifest.loc[batch, class_columns].to_numpy(np.int64).sum(axis=0)
        audit_rows.append(
            {
                "batch_index": batch_index,
                "n_patches": len(batch),
                "minimum_class_exposure_px": int(exposure.min()),
                **{
                    f"class_{code}_exposure_px": int(exposure[position])
                    for position, code in enumerate(sorted(class_map))
                },
            }
        )
    schedule = pd.DataFrame(rows)
    audit = pd.DataFrame(audit_rows)
    if schedule.manifest_index.nunique() != n_patches:
        raise AssertionError("Schedule does not use every selected patch exactly once")
    return schedule, audit


def prepare_data(
    bundle: ConfigBundle,
    data_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    data_root = Path(data_root).resolve()
    output = Path(output).resolve()
    training_file_specs = [
        item for item in bundle.dataset["files"]
        if str(item["role"]) in {"image", "label"}
    ]
    completion_path = output / "DATA_PREPARATION_COMPLETE.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("preparation_method_hash") != stable_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "implementation_revision": IMPLEMENTATION_REVISION,
                "dataset": bundle.dataset,
                "split": bundle.split,
            }
        ):
            raise ValueError("Existing prepared data belongs to a different dataset/split method")
        verify_files(data_root, training_file_specs)
        contract_path = output / "data_contract.json"
        if not contract_path.is_file() or sha256_file(contract_path) != completion.get("data_contract_sha256"):
            raise ValueError("Existing prepared data_contract.json hash mismatch")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        for role, filename in contract["artifact_files"].items():
            artifact_path = output / str(filename)
            expected = contract["artifact_sha256"].get(role)
            if not artifact_path.is_file() or sha256_file(artifact_path) != expected:
                raise ValueError(f"Existing prepared artifact failed verification: {role}")
        print(f"Prepared data already complete: {completion_path}")
        return completion
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Prepared-data output is non-empty but incomplete: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_verification = verify_files(data_root, training_file_specs)
    image_spec = _role_spec(bundle.dataset, "image")
    label_spec = _role_spec(bundle.dataset, "label")
    image_path = data_root / str(image_spec["path"])
    label_path = data_root / str(label_spec["path"])
    class_map = {int(code): str(name) for code, name in bundle.dataset["class_map"].items()}
    with rasterio.open(image_path) as image_source, rasterio.open(label_path) as label_source:
        if image_source.count != len(bundle.dataset["bands"]):
            raise ValueError("Image band count differs from dataset config")
        if image_source.shape != label_source.shape:
            raise ValueError("Image and label shapes differ")
        if image_source.crs != label_source.crs or not image_source.transform.almost_equals(label_source.transform):
            raise ValueError("Image and label must have identical CRS and affine transform")
        expected_shape = tuple(int(value) for value in bundle.dataset.get("expected_shape", image_source.shape))
        if image_source.shape != expected_shape:
            raise ValueError(f"Image shape {image_source.shape} differs from {expected_shape}")
        expected_crs = str(bundle.dataset.get("expected_crs", image_source.crs))
        if str(image_source.crs) != expected_crs:
            raise ValueError(f"Image CRS {image_source.crs} differs from {expected_crs}")
        image = image_source.read().astype(np.float32)
        label = label_source.read(1)
        nodata = image_source.nodata
        profile = image_source.profile
        transform = image_source.transform
        crs = image_source.crs

    image_valid = np.isfinite(image).all(axis=0)
    if nodata is not None:
        image_valid &= np.all(image != nodata, axis=0)
    common_valid = image_valid & (label >= 1) & (label <= len(class_map))
    domains = build_split_domains(image_valid.shape, common_valid, bundle.split)
    split_mask = domains["split_mask"]
    train_valid = common_valid & (split_mask == 1)
    validation_valid = common_valid & (split_mask == 2)
    if not train_valid.any() or not validation_valid.any():
        raise ValueError("Split produced an empty train or validation core")

    normalization_rows: list[dict[str, Any]] = []
    for band_index, band_name in enumerate(bundle.dataset["bands"]):
        values = image[band_index, train_valid].astype(np.float64)
        normalization_rows.append(
            {
                "band_index_1based": band_index + 1,
                "band": band_name,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "train_pixel_count": int(values.size),
            }
        )
    normalization = pd.DataFrame(normalization_rows)
    if np.any(normalization["std"].to_numpy() <= 0):
        raise ValueError("At least one image band has zero train-core standard deviation")

    manifest = _sample_manifest(
        label,
        common_valid,
        split_mask,
        domains["train_input"],
        bundle.split,
        class_map,
    )
    schedule, batch_audit = _build_schedule(manifest, bundle.split, class_map)
    manifest["batch_group_id"] = -1
    manifest["batch_slot"] = -1
    manifest["anchor_code"] = 0
    for row in schedule.itertuples(index=False):
        manifest.loc[int(row.manifest_index), ["batch_group_id", "batch_slot", "anchor_code"]] = [
            int(row.batch_index), int(row.slot), int(row.anchor_code)
        ]
    validation_manifest = _build_validation_manifest(
        image_valid.shape,
        bundle.split,
        image_valid,
        validation_valid,
        domains["validation_input"],
    )

    artifact_paths = {
        "image_valid_mask": output / "image_valid_mask.tif",
        "common_valid_mask": output / "common_valid_mask.tif",
        "split_mask": output / "split_mask.tif",
        "train_input_domain": output / "train_input_domain.tif",
        "validation_input_domain": output / "validation_input_domain.tif",
        "normalization": output / "normalization_train_only.csv",
        "train_manifest": output / "train_manifest.csv",
        "batch_schedule": output / "batch_schedule_pass0.csv",
        "batch_audit": output / "batch_class_audit.csv",
        "validation_manifest": output / "validation_inference_manifest.csv",
    }
    _atomic_raster(image_valid.astype(np.uint8), artifact_paths["image_valid_mask"], profile)
    _atomic_raster(common_valid.astype(np.uint8), artifact_paths["common_valid_mask"], profile)
    _atomic_raster(split_mask.astype(np.uint8), artifact_paths["split_mask"], profile)
    _atomic_raster(domains["train_input"].astype(np.uint8), artifact_paths["train_input_domain"], profile)
    _atomic_raster(domains["validation_input"].astype(np.uint8), artifact_paths["validation_input_domain"], profile)
    _atomic_csv(normalization, artifact_paths["normalization"])
    _atomic_csv(manifest, artifact_paths["train_manifest"])
    _atomic_csv(schedule, artifact_paths["batch_schedule"])
    _atomic_csv(batch_audit, artifact_paths["batch_audit"])
    _atomic_csv(validation_manifest, artifact_paths["validation_manifest"])

    preparation_payload = {
        "schema_version": SCHEMA_VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "dataset": bundle.dataset,
        "split": bundle.split,
    }
    preparation_method_hash = stable_hash(preparation_payload)
    artifact_hashes = {name: sha256_file(path) for name, path in artifact_paths.items()}
    train_counts = np.bincount(
        label[train_valid].astype(np.int64), minlength=len(class_map) + 1
    )[1:]
    contract = {
        "schema_version": SCHEMA_VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "preparation_method_hash": preparation_method_hash,
        "dataset_id": bundle.dataset["dataset_id"],
        "dataset_config_hash": stable_hash(bundle.dataset),
        "split_id": bundle.split["split_id"],
        "split_config_hash": stable_hash(bundle.split),
        "source_fingerprints": {
            str(item["role"]): str(item["sha256"]) for item in bundle.dataset["files"]
        },
        "grid": {
            "height": int(label.shape[0]),
            "width": int(label.shape[1]),
            "crs": str(crs),
            "transform": list(transform),
        },
        "pixel_counts": {
            "image_valid": int(image_valid.sum()),
            "common_valid": int(common_valid.sum()),
            "train_core": int(train_valid.sum()),
            "validation_core": int(validation_valid.sum()),
            "guard": int(np.count_nonzero(split_mask == 3)),
        },
        "class_counts_unique_train_core": train_counts.tolist(),
        "patch_contract": {
            "patch_size": int(bundle.split["patch_policy"]["size"]),
            "n_train_patches": len(manifest),
            "batch_size": int(bundle.split["sampling"]["batch_size"]),
            "batches_per_pass": int(len(schedule) // int(bundle.split["sampling"]["batch_size"])),
            "shared_train_validation_input_pixels": int(
                np.count_nonzero(domains["train_input"] & domains["validation_input"])
            ),
        },
        "artifact_files": {name: path.name for name, path in artifact_paths.items()},
        "artifact_sha256": artifact_hashes,
        "evaluation_contract": {
            "validation": "mean-softmax/probability mosaic on fixed spatial validation core",
            "verified_points_allowed_for_selection": False,
        },
    }
    contract_path = output / "data_contract.json"
    atomic_json(contract, contract_path)
    completion = {
        "status": "complete",
        "checkpoint": "DATA_PREPARATION",
        "created_at_unix": time.time(),
        "preparation_method_hash": preparation_method_hash,
        "data_contract_sha256": sha256_file(contract_path),
        "artifact_sha256": artifact_hashes,
        "n_train_patches": len(manifest),
        "n_batches": int(schedule.batch_index.nunique()),
        "validation_windows": len(validation_manifest),
        "shared_train_validation_input_pixels": contract["patch_contract"][
            "shared_train_validation_input_pixels"
        ],
        "source_verification": source_verification,
    }
    atomic_json(completion, completion_path)
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    return completion

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from ..contracts import sha256_file, stable_hash


REQUIRED_FILENAMES = (
    "CP2_COMPLETE.json",
    "data_contract.json",
    "image_valid_mask.tif",
    "common_valid_mask.tif",
    "split_mask_V_val1122_1389_guard48.tif",
    "train_manifest_4992.csv",
    "batch_schedule_pass0.csv",
    "validation_inference_manifest.csv",
    "normalization_train_only.csv",
)


def _role_path(dataset: dict[str, Any], data_root: Path, role: str) -> Path:
    matches = [item for item in dataset["files"] if str(item["role"]) == role]
    if len(matches) != 1:
        raise ValueError(f"Dataset contract must contain exactly one {role!r} file")
    return data_root / str(matches[0]["path"])


def verify_cp2_artifacts(
    artifact_root: str | Path,
    split_config: dict[str, Any],
) -> dict[str, Any]:
    """Verify the frozen Notebook-4 CP2 files against the repository contract."""
    root = Path(artifact_root).resolve()
    contract = split_config.get("artifact_contract", {})
    if contract.get("type") != "notebook4_cp2":
        raise ValueError("This runner requires split.artifact_contract.type=notebook4_cp2")
    expected = {str(name): str(value) for name, value in contract.get("files_sha256", {}).items()}
    missing_contract = sorted(set(REQUIRED_FILENAMES) - set(expected))
    if missing_contract:
        raise ValueError(f"Split config does not bind required CP2 files: {missing_contract}")

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for name in REQUIRED_FILENAMES:
        path = root / name
        if not path.is_file():
            rows.append({"file": name, "status": "missing"})
            failures.append(f"missing {path}")
            continue
        actual = sha256_file(path)
        status = "ok" if actual == expected[name] else "sha256_mismatch"
        rows.append({"file": name, "bytes": path.stat().st_size, "sha256": actual, "status": status})
        if status != "ok":
            failures.append(f"{name}: expected {expected[name]}, received {actual}")
    if failures:
        raise ValueError("CP2 artifact contract violation:\n- " + "\n- ".join(failures))

    completion = json.loads((root / "CP2_COMPLETE.json").read_text(encoding="utf-8"))
    data_contract = json.loads((root / "data_contract.json").read_text(encoding="utf-8"))
    expected_method_hash = str(contract["data_method_hash"])
    if completion.get("status") != "complete" or completion.get("checkpoint") != "CP2":
        raise ValueError("CP2_COMPLETE.json is not a completed CP2 marker")
    if completion.get("data_method_hash") != expected_method_hash:
        raise ValueError("CP2 completion data_method_hash differs from the split config")
    if data_contract.get("data_method_hash") != expected_method_hash:
        raise ValueError("CP2 data contract hash differs from the split config")
    return {"root": str(root), "files": rows, "completion": completion, "data_contract": data_contract}


def verify_prepared_artifacts(
    artifact_root: str | Path,
    dataset_config: dict[str, Any],
    split_config: dict[str, Any],
) -> dict[str, Any]:
    """Verify source-generated artifacts, falling back to the frozen Notebook-4 contract."""
    root = Path(artifact_root).resolve()
    completion_path = root / "DATA_PREPARATION_COMPLETE.json"
    if not completion_path.exists():
        legacy = verify_cp2_artifacts(root, split_config)
        legacy["artifact_files"] = {
            "image_valid_mask": "image_valid_mask.tif",
            "common_valid_mask": "common_valid_mask.tif",
            "split_mask": "split_mask_V_val1122_1389_guard48.tif",
            "normalization": "normalization_train_only.csv",
            "train_manifest": "train_manifest_4992.csv",
            "batch_schedule": "batch_schedule_pass0.csv",
            "validation_manifest": "validation_inference_manifest.csv",
        }
        legacy["mode"] = "legacy_notebook4_cp2"
        return legacy

    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    contract_path = root / "data_contract.json"
    if completion.get("status") != "complete" or completion.get("checkpoint") != "DATA_PREPARATION":
        raise ValueError("DATA_PREPARATION_COMPLETE.json is not a completed marker")
    if not contract_path.is_file() or sha256_file(contract_path) != completion.get("data_contract_sha256"):
        raise ValueError("Prepared data_contract.json hash mismatch")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("dataset_config_hash") != stable_hash(dataset_config):
        raise ValueError("Prepared artifacts were generated for a different dataset config")
    if contract.get("split_config_hash") != stable_hash(split_config):
        raise ValueError("Prepared artifacts were generated for a different split config")
    artifact_files = {str(key): str(value) for key, value in contract["artifact_files"].items()}
    artifact_hashes = {str(key): str(value) for key, value in contract["artifact_sha256"].items()}
    required_roles = {
        "image_valid_mask", "common_valid_mask", "split_mask", "normalization",
        "train_manifest", "batch_schedule", "validation_manifest",
    }
    missing_roles = sorted(required_roles - set(artifact_files))
    if missing_roles:
        raise ValueError(f"Prepared data contract is missing artifact roles: {missing_roles}")
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for role, filename in artifact_files.items():
        path = root / filename
        if not path.is_file():
            failures.append(f"missing {role}: {path}")
            continue
        actual = sha256_file(path)
        expected = artifact_hashes.get(role)
        status = "ok" if actual == expected else "sha256_mismatch"
        rows.append({"role": role, "file": filename, "sha256": actual, "status": status})
        if status != "ok":
            failures.append(f"{role}: expected {expected}, received {actual}")
    if failures:
        raise ValueError("Prepared artifact contract violation:\n- " + "\n- ".join(failures))
    return {
        "root": str(root),
        "files": rows,
        "completion": completion,
        "data_contract": contract,
        "artifact_files": artifact_files,
        "mode": "source_generated",
    }


class CachedScenePatchDataset(Dataset):
    def __init__(
        self,
        normalized_image: np.ndarray,
        target: np.ndarray,
        manifest: pd.DataFrame,
        patch_size: int,
    ) -> None:
        self.image = normalized_image
        self.target = target
        self.manifest = manifest.reset_index(drop=True).copy()
        self.patch_size = int(patch_size)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[int(index)]
        row0, col0 = int(row["row"]), int(row["col"])
        size = self.patch_size
        image = np.ascontiguousarray(self.image[:, row0 : row0 + size, col0 : col0 + size])
        target = np.ascontiguousarray(
            self.target[row0 : row0 + size, col0 : col0 + size],
            dtype=np.int64,
        )
        return {
            "image": torch.from_numpy(image),
            "target": torch.from_numpy(target),
            "patch_id": str(row["patch_id"]),
            "manifest_index": int(index),
        }


class ValidationWindowDataset(Dataset):
    def __init__(
        self,
        normalized_image: np.ndarray,
        valid_mask: np.ndarray,
        manifest: pd.DataFrame,
        patch_size: int,
    ) -> None:
        self.image = normalized_image
        self.valid_mask = valid_mask
        self.manifest = manifest.reset_index(drop=True).copy()
        self.patch_size = int(patch_size)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[int(index)]
        row0, col0 = int(row["row"]), int(row["col"])
        height, width = int(row["height"]), int(row["width"])
        image = np.zeros(
            (self.image.shape[0], self.patch_size, self.patch_size),
            dtype=self.image.dtype,
        )
        valid = np.zeros((self.patch_size, self.patch_size), dtype=bool)
        image[:, :height, :width] = self.image[:, row0 : row0 + height, col0 : col0 + width]
        valid[:height, :width] = self.valid_mask[row0 : row0 + height, col0 : col0 + width]
        return {
            "image": torch.from_numpy(image),
            "image_valid": torch.from_numpy(valid),
            "row": row0,
            "col": col0,
            "height": height,
            "width": width,
        }


@dataclass
class CP2Data:
    image: np.ndarray
    label: np.ndarray
    image_valid: np.ndarray
    common_valid: np.ndarray
    split_mask: np.ndarray
    target: np.ndarray
    train_manifest: pd.DataFrame
    batch_schedule: pd.DataFrame
    validation_manifest: pd.DataFrame
    frozen_batches: list[list[int]]
    train_dataset: CachedScenePatchDataset
    validation_dataset: ValidationWindowDataset
    class_map: dict[int, str]
    ignore_index: int
    contract: dict[str, Any]
    verification: dict[str, Any]

    def materialize_batch(self, batch_index: int) -> dict[str, Any]:
        indices = self.frozen_batches[int(batch_index)]
        return default_collate([self.train_dataset[index] for index in indices])


def load_cp2_data(
    dataset_config: dict[str, Any],
    split_config: dict[str, Any],
    data_root: str | Path,
    artifact_root: str | Path,
) -> CP2Data:
    data_root = Path(data_root).resolve()
    artifact_root = Path(artifact_root).resolve()
    verification = verify_prepared_artifacts(
        artifact_root,
        dataset_config,
        split_config,
    )
    contract = verification["data_contract"]
    files = verification["artifact_files"]

    image_path = _role_path(dataset_config, data_root, "image")
    label_path = _role_path(dataset_config, data_root, "label")
    expected_shape = tuple(int(value) for value in dataset_config.get("expected_shape", []))
    expected_crs = str(dataset_config.get("expected_crs", ""))
    class_map = {int(code): str(name) for code, name in dataset_config["class_map"].items()}
    num_classes = len(class_map)
    ignore_index = int(dataset_config.get("validity", {}).get("ignore_index", 255))
    patch_size = int(split_config["patch_policy"]["size"])

    with rasterio.open(image_path) as image_src, rasterio.open(label_path) as label_src:
        if image_src.count != len(dataset_config["bands"]):
            raise ValueError(f"Expected {len(dataset_config['bands'])} image bands, got {image_src.count}")
        if image_src.shape != label_src.shape:
            raise ValueError("Image and label shapes differ")
        if expected_shape and image_src.shape != expected_shape:
            raise ValueError(f"Raster shape {image_src.shape} differs from contract {expected_shape}")
        if expected_crs and str(image_src.crs) != expected_crs:
            raise ValueError(f"Raster CRS {image_src.crs} differs from contract {expected_crs}")
        if image_src.crs != label_src.crs or not image_src.transform.almost_equals(label_src.transform):
            raise ValueError("Image and label grids are not aligned")
        image = image_src.read().astype(np.float32)
        label = label_src.read(1)
        nodata = image_src.nodata

    with rasterio.open(artifact_root / files["image_valid_mask"]) as source:
        image_valid = source.read(1).astype(bool)
    with rasterio.open(artifact_root / files["common_valid_mask"]) as source:
        common_valid = source.read(1).astype(bool)
    with rasterio.open(artifact_root / files["split_mask"]) as source:
        split_mask = source.read(1).astype(np.uint8)

    computed_image_valid = np.isfinite(image).all(axis=0)
    if nodata is not None:
        computed_image_valid &= np.all(image != nodata, axis=0)
    computed_common_valid = computed_image_valid & (label >= 1) & (label <= num_classes)
    if not np.array_equal(computed_image_valid, image_valid):
        raise ValueError("image_valid_mask.tif does not match the source image")
    if not np.array_equal(computed_common_valid, common_valid):
        raise ValueError("common_valid_mask.tif does not match image/label validity")
    if not set(np.unique(split_mask)).issubset({0, 1, 2, 3}):
        raise ValueError("Unexpected split-mask code")

    normalization = pd.read_csv(artifact_root / files["normalization"])
    means = normalization["mean"].to_numpy(np.float32)[:, None, None]
    standard_deviations = normalization["std"].to_numpy(np.float32)[:, None, None]
    if means.shape != (image.shape[0], 1, 1) or np.any(standard_deviations <= 0):
        raise ValueError("Invalid train-only normalization artifact")
    image = (image - means) / standard_deviations
    image[:, ~image_valid] = 0.0
    if not np.isfinite(image).all():
        raise ValueError("Non-finite values remain after normalization")

    target = np.full(label.shape, ignore_index, dtype=np.uint8)
    train_valid = common_valid & (split_mask == 1)
    target[train_valid] = label[train_valid].astype(np.uint8) - 1
    expected_train_pixels = int(contract["pixel_counts"]["train_core"])
    if int(train_valid.sum()) != expected_train_pixels:
        raise ValueError("Train-core pixel count differs from CP2 contract")

    train_manifest = pd.read_csv(artifact_root / files["train_manifest"])
    schedule = pd.read_csv(artifact_root / files["batch_schedule"]).sort_values(
        ["batch_index", "slot"]
    ).reset_index(drop=True)
    validation_manifest = pd.read_csv(artifact_root / files["validation_manifest"])
    expected_patches = int(split_config["sampling"]["n_patches"])
    batch_size = int(split_config["sampling"]["batch_size"])
    if len(train_manifest) != expected_patches or len(schedule) != expected_patches:
        raise ValueError("Manifest/schedule length differs from the split config")
    frozen_batches = [
        group.sort_values("slot")["manifest_index"].astype(int).tolist()
        for _, group in schedule.groupby("batch_index", sort=True)
    ]
    if not frozen_batches or any(len(batch) != batch_size for batch in frozen_batches):
        raise ValueError("Frozen CP2 batches do not have the configured batch size")
    flat = [index for batch in frozen_batches for index in batch]
    if sorted(flat) != list(range(expected_patches)):
        raise ValueError("Frozen schedule does not use every manifest row exactly once")

    train_dataset = CachedScenePatchDataset(image, target, train_manifest, patch_size)
    validation_dataset = ValidationWindowDataset(
        image,
        image_valid,
        validation_manifest,
        patch_size,
    )
    return CP2Data(
        image=image,
        label=label,
        image_valid=image_valid,
        common_valid=common_valid,
        split_mask=split_mask,
        target=target,
        train_manifest=train_manifest,
        batch_schedule=schedule,
        validation_manifest=validation_manifest,
        frozen_batches=frozen_batches,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        class_map=class_map,
        ignore_index=ignore_index,
        contract=contract,
        verification=verification,
    )

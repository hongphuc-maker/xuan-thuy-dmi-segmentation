from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
from tqdm.auto import tqdm

from .config import ConfigBundle, TemporalTransferBundle
from .contracts import atomic_json, sha256_file, stable_hash, verify_files
from .data.cp2 import verify_prepared_artifacts
from .independent_evaluation import evaluate_verified_points
from .inference import _starts, _torch_load_cpu
from .models import build_model
from .reproducibility import configure_reproducibility


def _single_role(dataset: dict[str, Any], role: str) -> dict[str, Any]:
    matches = [item for item in dataset["files"] if str(item["role"]) == role]
    if len(matches) != 1:
        raise ValueError(f"Dataset must define exactly one {role!r} file")
    return matches[0]


def _evaluation_bundle(bundle: TemporalTransferBundle) -> ConfigBundle:
    experiment = {
        "experiment_id": bundle.transfer["transfer_id"],
        "evaluation_interpretation": bundle.transfer["evaluation_interpretation"],
    }
    resolved = copy.deepcopy(bundle.resolved)
    resolved["method_hash"] = bundle.transfer_hash
    return ConfigBundle(
        experiment_path=bundle.transfer_path,
        experiment=experiment,
        dataset=bundle.target.dataset,
        split=bundle.target.split,
        model=bundle.source.model,
        losses={},
        resolved=resolved,
        method_hash=bundle.transfer_hash,
    )


def infer_temporal_transfer_map(
    bundle: TemporalTransferBundle,
    data_root: str | Path,
    source_artifact_root: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Apply a sealed source model to a target-date scene using source normalization."""
    data_root = Path(data_root).resolve()
    source_artifact_root = Path(source_artifact_root).resolve()
    checkpoint = Path(checkpoint).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    map_path = output / "prediction.tif"
    completion_path = output / "MAP_COMPLETE.json"
    checkpoint_sha = sha256_file(checkpoint)
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("transfer_hash") != bundle.transfer_hash:
            raise ValueError("Existing map belongs to a different temporal-transfer contract")
        if completion.get("source_checkpoint_sha256") != checkpoint_sha:
            raise ValueError("Existing map belongs to a different source checkpoint")
        if not map_path.is_file() or sha256_file(map_path) != completion.get("map_sha256"):
            raise ValueError("Existing transfer map differs from MAP_COMPLETE.json")
        print(f"Temporal-transfer map already complete: {map_path}")
        return completion

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    runtime = configure_reproducibility(
        int(bundle.source.experiment["seed"]), warn_only=False
    )

    target_image_spec = _single_role(bundle.target.dataset, "image")
    target_verification = verify_files(data_root, [target_image_spec])
    source_artifacts = verify_prepared_artifacts(
        source_artifact_root,
        bundle.source.dataset,
        bundle.source.split,
    )
    normalization_path = source_artifact_root / source_artifacts["artifact_files"]["normalization"]
    normalization = pd.read_csv(normalization_path)

    target_image_path = data_root / str(target_image_spec["path"])
    with rasterio.open(target_image_path) as source:
        if source.count != len(bundle.target.dataset["bands"]):
            raise ValueError("Target image band count differs from the target dataset contract")
        expected_shape = tuple(int(value) for value in bundle.target.dataset.get("expected_shape", []))
        if expected_shape and source.shape != expected_shape:
            raise ValueError(
                f"Target raster shape {source.shape} differs from contract {expected_shape}"
            )
        expected_crs = str(bundle.target.dataset.get("expected_crs", ""))
        if expected_crs and str(source.crs) != expected_crs:
            raise ValueError(f"Target raster CRS {source.crs} differs from contract {expected_crs}")
        raw_image = source.read().astype(np.float32)
        nodata = source.nodata
        profile = source.profile.copy()

    image_valid = np.isfinite(raw_image).all(axis=0)
    if nodata is not None:
        image_valid &= np.all(raw_image != nodata, axis=0)
    means = normalization["mean"].to_numpy(np.float32)[:, None, None]
    standard_deviations = normalization["std"].to_numpy(np.float32)[:, None, None]
    if means.shape != (raw_image.shape[0], 1, 1) or np.any(standard_deviations <= 0):
        raise ValueError("Invalid source-training normalization artifact")
    image = (raw_image - means) / standard_deviations
    image[:, ~image_valid] = 0.0
    if not np.isfinite(image).all():
        raise ValueError("Non-finite values remain after source normalization")

    training_completion_path = checkpoint.parent / "TRAINING_COMPLETE.json"
    if not training_completion_path.is_file():
        raise ValueError("Temporal transfer requires a sealed TRAINING_COMPLETE.json")
    training_completion = json.loads(training_completion_path.read_text(encoding="utf-8"))
    if training_completion.get("method_hash") != bundle.source.method_hash:
        raise ValueError("Source training completion and source experiment hash differ")
    if training_completion.get("final_model_sha256") != checkpoint_sha:
        raise ValueError("Source checkpoint differs from TRAINING_COMPLETE.json")
    payload = _torch_load_cpu(checkpoint)
    if payload.get("method_hash") != bundle.source.method_hash:
        raise ValueError("Source checkpoint and source experiment hash differ")
    if payload.get("model_config") and payload["model_config"] != bundle.source.model:
        raise ValueError("Source checkpoint model config differs from transfer model config")

    model = build_model(bundle.source.model).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()

    patch = int(bundle.target.split["patch_policy"]["size"])
    stride = int(bundle.target.split["patch_policy"].get("inference_stride", patch // 2))
    batch_size = int(bundle.target.split["sampling"]["batch_size"])
    height, width = image_valid.shape
    windows = [
        (row, col)
        for row in _starts(height, patch, stride)
        for col in _starts(width, patch, stride)
    ]
    num_classes = len(bundle.target.dataset["class_map"])
    probability_sum = np.zeros((num_classes, height, width), dtype=np.float32)
    probability_count = np.zeros((height, width), dtype=np.uint16)
    output_type = str(bundle.source.model.get("output", "logits"))
    with torch.inference_mode():
        progress = tqdm(range(0, len(windows), batch_size), desc="temporal-transfer", unit="batch")
        for start in progress:
            batch_windows = windows[start : start + batch_size]
            arrays: list[np.ndarray] = []
            shapes: list[tuple[int, int]] = []
            for row, col in batch_windows:
                height_here = min(patch, height - row)
                width_here = min(patch, width - col)
                padded = np.zeros((image.shape[0], patch, patch), dtype=image.dtype)
                padded[:, :height_here, :width_here] = image[
                    :, row : row + height_here, col : col + width_here
                ]
                arrays.append(padded)
                shapes.append((height_here, width_here))
            inputs = torch.from_numpy(np.stack(arrays)).to(device, non_blocking=True)
            outputs = model(inputs)
            if output_type == "logits":
                outputs = torch.softmax(outputs, dim=1)
            elif output_type != "probabilities":
                raise ValueError(f"Unsupported model output type: {output_type}")
            probabilities = outputs.cpu().numpy()
            for index, (row, col) in enumerate(batch_windows):
                height_here, width_here = shapes[index]
                valid = image_valid[row : row + height_here, col : col + width_here]
                probability_sum[:, row : row + height_here, col : col + width_here] += (
                    probabilities[index, :, :height_here, :width_here] * valid[None]
                )
                probability_count[row : row + height_here, col : col + width_here] += (
                    valid.astype(np.uint16)
                )

    if not np.all(probability_count[image_valid] > 0):
        raise ValueError("Transfer windows did not cover every valid target pixel")
    prediction = np.zeros((height, width), dtype=np.uint8)
    averaged = probability_sum[:, image_valid] / probability_count[image_valid][None]
    prediction[image_valid] = averaged.argmax(axis=0).astype(np.uint8) + 1

    profile.update(
        driver="GTiff", count=1, dtype="uint8", nodata=0,
        compress="deflate", predictor=1,
    )
    temporary = map_path.with_name(map_path.stem + ".tmp" + map_path.suffix)
    with rasterio.open(temporary, "w", **profile) as destination:
        destination.write(prediction, 1)
        destination.update_tags(
            transfer_id=str(bundle.transfer["transfer_id"]),
            transfer_hash=bundle.transfer_hash,
            method_hash=bundle.transfer_hash,
            source_experiment_id=str(bundle.source.experiment["experiment_id"]),
            source_method_hash=bundle.source.method_hash,
            source_checkpoint_sha256=checkpoint_sha,
            target_dataset_id=str(bundle.target.dataset["dataset_id"]),
            normalization_policy="source_training_artifact",
            class_codes="1..11",
        )
    os.replace(temporary, map_path)

    completion = {
        "status": "complete",
        "checkpoint": "TEMPORAL_TRANSFER_MAP",
        "transfer_id": bundle.transfer["transfer_id"],
        "experiment_id": bundle.transfer["transfer_id"],
        "transfer_hash": bundle.transfer_hash,
        "method_hash": bundle.transfer_hash,
        "source_experiment_id": bundle.source.experiment["experiment_id"],
        "source_method_hash": bundle.source.method_hash,
        "source_dataset_id": bundle.source.dataset["dataset_id"],
        "target_dataset_id": bundle.target.dataset["dataset_id"],
        "normalization_policy": "source_training_artifact",
        "source_normalization_sha256": sha256_file(normalization_path),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": checkpoint_sha,
        "source_training_completion_sha256": sha256_file(training_completion_path),
        "target_image_sha256": str(target_image_spec["sha256"]),
        "map": map_path.name,
        "map_sha256": sha256_file(map_path),
        "valid_output_pixels": int(np.count_nonzero(prediction)),
        "predicted_classes": sorted(int(value) for value in np.unique(prediction[prediction > 0])),
        "evaluation_interpretation": bundle.transfer["evaluation_interpretation"],
        "source_artifact_contract_hash": stable_hash(source_artifacts["data_contract"]),
        "target_verification": target_verification,
        "runtime": runtime,
        "verified_points_used": False,
    }
    atomic_json(completion, completion_path)
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    return completion


def evaluate_temporal_transfer_points(
    bundle: TemporalTransferBundle,
    data_root: str | Path,
    map_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    result = evaluate_verified_points(
        _evaluation_bundle(bundle),
        data_root,
        map_path,
        output,
    )
    enriched = {
        **result,
        "checkpoint": "TEMPORAL_TRANSFER_POINT_EVALUATION",
        "transfer_id": bundle.transfer["transfer_id"],
        "transfer_hash": bundle.transfer_hash,
        "source_experiment_id": bundle.source.experiment["experiment_id"],
        "source_dataset_id": bundle.source.dataset["dataset_id"],
        "target_dataset_id": bundle.target.dataset["dataset_id"],
        "normalization_policy": bundle.transfer["normalization_policy"],
        "evaluation_interpretation": bundle.transfer["evaluation_interpretation"],
    }
    atomic_json(
        enriched,
        Path(output).resolve() / "INDEPENDENT_EVALUATION_COMPLETE.json",
    )
    return enriched

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from tqdm.auto import tqdm

from .config import ConfigBundle
from .contracts import atomic_json, sha256_file, verify_files
from .data.cp2 import load_cp2_data
from .models import build_model
from .reproducibility import configure_reproducibility


def _torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _starts(extent: int, size: int, stride: int) -> list[int]:
    if extent <= size:
        return [0]
    values = list(range(0, extent - size + 1, stride))
    if values[-1] != extent - size:
        values.append(extent - size)
    return values


def infer_full_map(
    bundle: ConfigBundle,
    data_root: str | Path,
    artifact_root: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    device_name: str = "cuda",
) -> dict[str, Any]:
    data_root = Path(data_root).resolve()
    artifact_root = Path(artifact_root).resolve()
    checkpoint = Path(checkpoint).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    map_path = output / "prediction.tif"
    completion_path = output / "MAP_COMPLETE.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("method_hash") != bundle.method_hash:
            raise ValueError("Existing map belongs to a different experiment method hash")
        if sha256_file(map_path) != completion.get("map_sha256"):
            raise ValueError("Existing prediction map differs from MAP_COMPLETE.json")
        print(f"Map already complete: {map_path}")
        return completion

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    configure_reproducibility(int(bundle.experiment["seed"]), warn_only=False)
    source_verification = verify_files(
        data_root,
        [item for item in bundle.dataset["files"] if str(item["role"]) in {"image", "label"}],
    )
    data = load_cp2_data(bundle.dataset, bundle.split, data_root, artifact_root)
    training_completion_path = checkpoint.parent / "TRAINING_COMPLETE.json"
    if not training_completion_path.is_file():
        raise ValueError("Final-map inference requires a sealed TRAINING_COMPLETE.json")
    training_completion = json.loads(training_completion_path.read_text(encoding="utf-8"))
    if training_completion.get("method_hash") != bundle.method_hash:
        raise ValueError("Training completion and experiment method hashes differ")
    if training_completion.get("final_model_sha256") != sha256_file(checkpoint):
        raise ValueError("Checkpoint differs from TRAINING_COMPLETE.json")
    payload = _torch_load_cpu(checkpoint)
    if payload.get("method_hash") != bundle.method_hash:
        raise ValueError("Checkpoint and experiment method hashes differ")
    if payload.get("model_config") and payload["model_config"] != bundle.model:
        raise ValueError("Checkpoint model config differs from experiment model config")
    model = build_model(bundle.model).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()

    patch = int(bundle.split["patch_policy"]["size"])
    stride = int(bundle.split["patch_policy"].get("inference_stride", patch // 2))
    height, width = data.label.shape
    windows = [(row, col) for row in _starts(height, patch, stride) for col in _starts(width, patch, stride)]
    num_classes = len(data.class_map)
    probability_sum = np.zeros((num_classes, height, width), dtype=np.float32)
    probability_count = np.zeros((height, width), dtype=np.uint16)
    batch_size = int(bundle.split["sampling"]["batch_size"])
    output_type = str(bundle.model.get("output", "logits"))
    with torch.inference_mode():
        progress = tqdm(range(0, len(windows), batch_size), desc="full-map", unit="batch")
        for start in progress:
            batch_windows = windows[start : start + batch_size]
            arrays: list[np.ndarray] = []
            shapes: list[tuple[int, int]] = []
            for row, col in batch_windows:
                height_here = min(patch, height - row)
                width_here = min(patch, width - col)
                padded = np.zeros((data.image.shape[0], patch, patch), dtype=data.image.dtype)
                padded[:, :height_here, :width_here] = data.image[
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
                valid = data.image_valid[row : row + height_here, col : col + width_here]
                probability_sum[:, row : row + height_here, col : col + width_here] += (
                    probabilities[index, :, :height_here, :width_here] * valid[None]
                )
                probability_count[row : row + height_here, col : col + width_here] += valid.astype(np.uint16)

    if not np.all(probability_count[data.image_valid] > 0):
        raise ValueError("Full-map windows did not cover every valid image pixel")
    prediction = np.zeros((height, width), dtype=np.uint8)
    averaged = probability_sum[:, data.image_valid] / probability_count[data.image_valid][None]
    prediction[data.image_valid] = averaged.argmax(axis=0).astype(np.uint8) + 1

    image_spec = next(item for item in bundle.dataset["files"] if item["role"] == "image")
    with rasterio.open(data_root / str(image_spec["path"])) as source:
        profile = source.profile.copy()
    profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="deflate", predictor=1)
    temporary = map_path.with_name(map_path.stem + ".tmp" + map_path.suffix)
    with rasterio.open(temporary, "w", **profile) as destination:
        destination.write(prediction, 1)
        destination.update_tags(
            experiment_id=str(bundle.experiment["experiment_id"]),
            method_hash=bundle.method_hash,
            class_codes="1..11",
            checkpoint_sha256=sha256_file(checkpoint),
        )
    os.replace(temporary, map_path)
    completion = {
        "status": "complete",
        "checkpoint": "FULL_MAP_INFERENCE",
        "experiment_id": bundle.experiment["experiment_id"],
        "method_hash": bundle.method_hash,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "training_completion_sha256": sha256_file(training_completion_path),
        "map": map_path.name,
        "map_sha256": sha256_file(map_path),
        "valid_output_pixels": int(np.count_nonzero(prediction)),
        "predicted_classes": sorted(int(value) for value in np.unique(prediction[prediction > 0])),
        "source_verification": source_verification,
        "verified_points_used": False,
    }
    atomic_json(completion, completion_path)
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    return completion

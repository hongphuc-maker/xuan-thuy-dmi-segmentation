from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio

from .config import ConfigBundle
from .contracts import atomic_json, sha256_file


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        resolved = (destination / member.filename).resolve()
        if destination != resolved and destination not in resolved.parents:
            raise ValueError(f"Unsafe path in verified-point archive: {member.filename}")
    archive.extractall(destination)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _metrics(confusion: np.ndarray, class_map: dict[int, str]) -> tuple[dict[str, Any], pd.DataFrame]:
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(true_positive), where=(precision + recall) > 0)
    verified = support > 0
    summary = {
        "evaluated_points": int(confusion.sum()),
        "verified_classes": int(verified.sum()),
        "OA": float(true_positive.sum() / max(confusion.sum(), 1)),
        "macro_F1_verified_classes": float(f1[verified].mean()),
        "macro_recall_verified_classes": float(recall[verified].mean()),
        "predicted_classes_all_11": int((predicted > 0).sum()),
    }
    codes = sorted(class_map)
    per_class = pd.DataFrame(
        {
            "code": codes,
            "class": [class_map[code] for code in codes],
            "has_verified_ground_truth": verified,
            "support": support.astype(np.int64),
            "predicted": predicted.astype(np.int64),
            "precision": precision,
            "recall": recall,
            "F1": np.where(verified, f1, np.nan),
        }
    )
    return summary, per_class


def evaluate_verified_points(
    bundle: ConfigBundle,
    data_root: str | Path,
    map_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    try:
        import geopandas as gpd
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("Point evaluation requires geopandas and pyogrio") from error
    data_root = Path(data_root).resolve()
    map_path = Path(map_path).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    completion_path = output / "INDEPENDENT_EVALUATION_COMPLETE.json"
    if completion_path.exists():
        result = json.loads(completion_path.read_text(encoding="utf-8"))
        if result.get("method_hash") != bundle.method_hash:
            raise ValueError("Existing independent evaluation has a different method hash")
        if result.get("map_sha256") != sha256_file(map_path):
            raise ValueError("Existing evaluation belongs to a different map")
        for filename, expected in result.get("artifact_sha256", {}).items():
            path = output / filename
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"Independent-evaluation artifact failed verification: {filename}")
        return result

    map_completion_path = map_path.parent / "MAP_COMPLETE.json"
    if not map_completion_path.is_file():
        raise ValueError("Independent evaluation requires a sealed MAP_COMPLETE.json")
    map_completion = json.loads(map_completion_path.read_text(encoding="utf-8"))
    if map_completion.get("map_sha256") != sha256_file(map_path):
        raise ValueError("Map differs from its completion marker")
    if map_completion.get("method_hash") != bundle.method_hash:
        raise ValueError("Map and experiment method hashes differ")

    point_spec = next(
        (item for item in bundle.dataset["files"] if item["role"] == "verified_points"),
        None,
    )
    if point_spec is None:
        raise ValueError("Dataset config does not define verified_points")
    point_zip = data_root / str(point_spec["path"])
    if sha256_file(point_zip) != str(point_spec["sha256"]):
        raise ValueError("Verified-point archive SHA-256 mismatch")
    class_map = {int(code): str(name) for code, name in bundle.dataset["class_map"].items()}
    code_by_name = {name.casefold(): code for code, name in class_map.items()}

    with rasterio.open(map_path) as source:
        prediction = source.read(1)
        transform, map_crs, nodata = source.transform, source.crs, source.nodata
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="xtseg_verified_points_") as temporary:
        with zipfile.ZipFile(point_zip) as archive:
            _safe_extract(archive, Path(temporary))
        shapefiles = sorted(Path(temporary).rglob("*.shp"))
        if not shapefiles:
            raise ValueError("Verified-point ZIP contains no shapefiles")
        for shapefile in shapefiles:
            name = shapefile.stem.casefold()
            if name not in code_by_name:
                raise ValueError(f"Cannot map shapefile layer {shapefile.stem!r} to a class")
            truth = int(code_by_name[name])
            points = gpd.read_file(shapefile)
            if points.crs is None:
                raise ValueError(f"Point layer has no CRS: {shapefile.name}")
            points = points.to_crs(map_crs)
            if not points.geometry.notna().all() or not points.geometry.geom_type.eq("Point").all():
                raise ValueError(f"Layer must contain Point geometries only: {shapefile.name}")
            for source_index, feature in points.iterrows():
                row, col = rasterio.transform.rowcol(transform, feature.geometry.x, feature.geometry.y)
                inside = 0 <= row < prediction.shape[0] and 0 <= col < prediction.shape[1]
                predicted = int(prediction[row, col]) if inside else int(nodata or 0)
                records.append(
                    {
                        "layer": shapefile.stem,
                        "source_index": str(source_index),
                        "truth_code": truth,
                        "predicted_code": predicted,
                        "row": int(row),
                        "col": int(col),
                        "x": float(feature.geometry.x),
                        "y": float(feature.geometry.y),
                        "map_valid": bool(inside and predicted != int(nodata or 0)),
                    }
                )
    points = pd.DataFrame(records)
    evaluated = points[points.map_valid].copy()
    if evaluated.empty:
        raise ValueError("No verified points intersect valid map predictions")
    n_classes = len(class_map)
    invalid_codes = evaluated.loc[
        ~evaluated.predicted_code.between(1, n_classes), "predicted_code"
    ].unique()
    if len(invalid_codes):
        raise ValueError(f"Map contains prediction codes outside 1..{n_classes}: {invalid_codes}")
    confusion = np.bincount(
        (evaluated.truth_code.to_numpy(np.int64) - 1) * n_classes
        + (evaluated.predicted_code.to_numpy(np.int64) - 1),
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes)
    summary, per_class = _metrics(confusion, class_map)
    confusion_long = pd.DataFrame(
        [
            {"truth_code": truth + 1, "predicted_code": predicted + 1, "count": int(confusion[truth, predicted])}
            for truth in range(n_classes) for predicted in range(n_classes)
        ]
    )
    artifact_paths = {
        "verified_point_predictions.csv": output / "verified_point_predictions.csv",
        "verified_per_class_metrics.csv": output / "verified_per_class_metrics.csv",
        "verified_confusion_long.csv": output / "verified_confusion_long.csv",
    }
    _atomic_csv(points, artifact_paths["verified_point_predictions.csv"])
    _atomic_csv(per_class, artifact_paths["verified_per_class_metrics.csv"])
    _atomic_csv(confusion_long, artifact_paths["verified_confusion_long.csv"])
    result = {
        "status": "complete",
        "checkpoint": "INDEPENDENT_POINT_EVALUATION",
        "experiment_id": bundle.experiment["experiment_id"],
        "method_hash": bundle.method_hash,
        "map_sha256": sha256_file(map_path),
        "verified_points_sha256": sha256_file(point_zip),
        "input_points": len(points),
        "map_nodata_or_outside_points": int((~points.map_valid).sum()),
        **summary,
        "metric_scope_note": "macro_F1_verified_classes covers only classes present in verified ground truth",
        "selection_or_tuning_performed_from_verified_points": False,
        "verified_points_used": True,
        "artifact_sha256": {
            filename: sha256_file(path) for filename, path in artifact_paths.items()
        },
    }
    atomic_json(result, completion_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result

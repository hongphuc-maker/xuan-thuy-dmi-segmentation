from __future__ import annotations

from pathlib import Path
from typing import Any

import rasterio

from ..contracts import sha256_file


def inspect_inputs(
    data_root: str | Path,
    image: str | Path,
    label: str | Path,
    verified_points: str | Path | None = None,
    expected_bands: int = 10,
) -> dict[str, Any]:
    """Inspect an analysis-ready image/label pair before writing dataset YAML."""
    root = Path(data_root).resolve()
    image_path = (root / image).resolve()
    label_path = (root / label).resolve()
    for path in (image_path, label_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    with rasterio.open(image_path) as image_source, rasterio.open(label_path) as label_source:
        aligned = (
            image_source.shape == label_source.shape
            and image_source.crs == label_source.crs
            and image_source.transform.almost_equals(label_source.transform)
        )
        result: dict[str, Any] = {
            "status": "pass" if aligned and image_source.count == int(expected_bands) else "fail",
            "analysis_ready_contract": {
                "expected_bands": int(expected_bands),
                "actual_bands": int(image_source.count),
                "aligned_shape_crs_transform": bool(aligned),
            },
            "image": {
                "path": str(image),
                "sha256": sha256_file(image_path),
                "shape": list(image_source.shape),
                "count": int(image_source.count),
                "dtype": list(image_source.dtypes),
                "nodata": image_source.nodata,
                "crs": str(image_source.crs),
                "transform": list(image_source.transform),
            },
            "label": {
                "path": str(label),
                "sha256": sha256_file(label_path),
                "shape": list(label_source.shape),
                "count": int(label_source.count),
                "dtype": list(label_source.dtypes),
                "nodata": label_source.nodata,
                "crs": str(label_source.crs),
                "transform": list(label_source.transform),
            },
        }
    if verified_points is not None:
        points_path = (root / verified_points).resolve()
        if not points_path.is_file():
            raise FileNotFoundError(points_path)
        result["verified_points"] = {
            "path": str(verified_points),
            "sha256": sha256_file(points_path),
        }
    if result["status"] != "pass":
        raise ValueError(f"Inputs do not satisfy the analysis-ready contract: {result}")
    return result

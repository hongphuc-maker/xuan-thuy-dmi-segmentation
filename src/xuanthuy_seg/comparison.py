from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .contracts import atomic_json, sha256_file, stable_hash


_POINT_KEY = ("layer", "source_index", "truth_code", "row", "col")


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(_require_file(path).read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with _require_file(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"Missing numeric value {key!r}")
    return float(value)


def _int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def _atomic_csv(rows: Iterable[dict[str, Any]], path: Path) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Refusing to write an empty comparison table to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fieldnames = list(materialized[0])
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)
    os.replace(temporary, path)


def _phase_directory(run_root: Path) -> Path:
    phases = sorted(
        path
        for path in (run_root / "training" / "phases").iterdir()
        if (path / "validation_history.csv").is_file()
    )
    if len(phases) != 1:
        raise ValueError(
            f"Expected exactly one completed phase under {run_root}; found {len(phases)}"
        )
    return phases[0]


def _history_summary(run_root: Path) -> dict[str, Any]:
    completion = _load_json(run_root / "training" / "TRAINING_COMPLETE.json")
    if completion.get("status") != "complete":
        raise ValueError(f"Training is not complete under {run_root}")
    phase_completion = completion["phases"][0]
    history = _read_csv(_phase_directory(run_root) / "validation_history.csv")
    if not history:
        raise ValueError(f"Validation history is empty under {run_root}")

    selected_step = int(phase_completion["best_accepted_step"])
    selected = next(
        (row for row in history if _int(row, "optimizer_step") == selected_step),
        None,
    )
    if selected is None:
        raise ValueError(f"Selected step {selected_step} is absent from validation history")

    best_f1 = max(history, key=lambda row: _float(row, "validation_macro_f1_11"))
    best_iou = max(history, key=lambda row: _float(row, "validation_macro_iou_11"))
    best_validation_dmi = min(
        history,
        key=lambda row: _float(row, "validation_dmi_loss"),
    )

    def semantic(row: dict[str, str]) -> dict[str, Any]:
        return {
            "step": _int(row, "optimizer_step"),
            "oa": _float(row, "validation_oa"),
            "macro_f1_11": _float(row, "validation_macro_f1_11"),
            "macro_iou_11": _float(row, "validation_macro_iou_11"),
            "validation_dmi_loss": _float(row, "validation_dmi_loss"),
            "fixed_train_panel_dmi_loss": _float(
                row,
                "fixed_train_panel_dmi_loss"
                if row.get("fixed_train_panel_dmi_loss", "") != ""
                else "fixed_panel_dmi_loss",
            ),
            "predicted_classes": _int(row, "predicted_classes"),
        }

    return {
        "experiment_id": completion["experiment_id"],
        "method_hash": completion["method_hash"],
        "end_step": int(completion["end_global_step"]),
        "selected": semantic(selected),
        "best_macro_f1": semantic(best_f1),
        "best_macro_iou": semantic(best_iou),
        "minimum_validation_dmi": semantic(best_validation_dmi),
        "history": history,
        "final_model_sha256": completion["final_model_sha256"],
        "selected_checkpoint_sha256": phase_completion[
            "selected_checkpoint_sha256"
        ],
    }


def _matched_protocol(run_root: Path) -> dict[str, Any]:
    config = _load_json(run_root / "training" / "resolved_config.json")
    if len(config["phases"]) != 1:
        raise ValueError("Matched comparison requires one-phase DMI continuations")
    phase = config["phases"][0]
    initialization = phase.get("initialization", {})
    return {
        "seed": config["seed"],
        "dataset_hash": stable_hash(config["dataset"]),
        "split_hash": stable_hash(config["split"]),
        "loss_hash": stable_hash(phase["loss"]),
        "optimizer": phase["optimizer"],
        "maximum_steps": phase["maximum_steps"],
        "validation_interval_steps": phase["validation_interval_steps"],
        "monitor_interval_steps": phase["monitor_interval_steps"],
        "initialization_sha256": initialization["sha256"],
        "selection": config["selection"],
        "acceptance": config["acceptance"],
        "budget": config["budget"],
    }


def _assert_matching_grid(control_source: Any, morphology_source: Any) -> None:
    checks = {
        "width": (control_source.width, morphology_source.width),
        "height": (control_source.height, morphology_source.height),
        "count": (control_source.count, morphology_source.count),
        "crs": (control_source.crs, morphology_source.crs),
        "transform": (control_source.transform, morphology_source.transform),
    }
    failures = [key for key, values in checks.items() if values[0] != values[1]]
    if failures:
        raise ValueError(f"Map grid mismatch for: {', '.join(failures)}")
    if control_source.count != 1:
        raise ValueError("Comparison expects one-band class-code maps")


def _boundary_pixels(classes: np.ndarray, valid: np.ndarray) -> np.ndarray:
    boundary = np.zeros(classes.shape, dtype=bool)
    horizontal = (
        valid[:, 1:]
        & valid[:, :-1]
        & (classes[:, 1:] != classes[:, :-1])
    )
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    vertical = (
        valid[1:, :]
        & valid[:-1, :]
        & (classes[1:, :] != classes[:-1, :])
    )
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def _map_complexity(
    classes: np.ndarray,
    valid: np.ndarray,
    class_codes: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from scipy import ndimage

    structure = np.ones((3, 3), dtype=np.uint8)
    rows: list[dict[str, Any]] = []
    for code in class_codes:
        mask = valid & (classes == code)
        labels, components = ndimage.label(mask, structure=structure)
        sizes = np.bincount(labels.ravel())[1:]
        rows.append(
            {
                "code": code,
                "pixels": int(mask.sum()),
                "components_8_connected": int(components),
                "singleton_components": int(np.count_nonzero(sizes == 1)),
            }
        )
    return (
        {
            "valid_pixels": int(valid.sum()),
            "components_8_connected": sum(
                row["components_8_connected"] for row in rows
            ),
            "singleton_components": sum(row["singleton_components"] for row in rows),
            "boundary_pixels_4_neighbor": int(_boundary_pixels(classes, valid).sum()),
        },
        rows,
    )


def _relative_change(control: int, morphology: int) -> float | None:
    if control == 0:
        return None
    return (morphology - control) / control


def _compare_maps(
    control_path: Path,
    morphology_path: Path,
    output: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import rasterio

    with rasterio.open(_require_file(control_path)) as control_source, rasterio.open(
        _require_file(morphology_path)
    ) as morphology_source:
        _assert_matching_grid(control_source, morphology_source)
        grid = {
            "height": int(control_source.height),
            "width": int(control_source.width),
            "crs": str(control_source.crs),
            "transform": list(control_source.transform)[:6],
        }
        control = control_source.read(1)
        morphology = morphology_source.read(1)
        valid = (control_source.read_masks(1) > 0) & (
            morphology_source.read_masks(1) > 0
        )
        class_codes = sorted(
            set(np.unique(control[valid]).astype(int).tolist())
            | set(np.unique(morphology[valid]).astype(int).tolist())
        )
        control_summary, control_rows = _map_complexity(control, valid, class_codes)
        morphology_summary, morphology_rows = _map_complexity(
            morphology,
            valid,
            class_codes,
        )
        changed = valid & (control != morphology)

        changed_profile = control_source.profile.copy()
        changed_profile.update(dtype="uint8", nodata=255, compress="lzw")
        changed_raster = np.full(control.shape, 255, dtype=np.uint8)
        changed_raster[valid] = changed[valid].astype(np.uint8)
        changed_path = output / "matched_map_changed_pixels.tif"
        with rasterio.open(changed_path, "w", **changed_profile) as destination:
            destination.write(changed_raster, 1)

    control_by_code = {row["code"]: row for row in control_rows}
    morphology_by_code = {row["code"]: row for row in morphology_rows}
    per_class: list[dict[str, Any]] = []
    for code in class_codes:
        control_row = control_by_code[code]
        morphology_row = morphology_by_code[code]
        per_class.append(
            {
                "code": code,
                "control_pixels": control_row["pixels"],
                "morphology_pixels": morphology_row["pixels"],
                "pixel_delta": morphology_row["pixels"] - control_row["pixels"],
                "control_components_8_connected": control_row[
                    "components_8_connected"
                ],
                "morphology_components_8_connected": morphology_row[
                    "components_8_connected"
                ],
                "components_relative_change": _relative_change(
                    control_row["components_8_connected"],
                    morphology_row["components_8_connected"],
                ),
                "control_singleton_components": control_row[
                    "singleton_components"
                ],
                "morphology_singleton_components": morphology_row[
                    "singleton_components"
                ],
                "singletons_relative_change": _relative_change(
                    control_row["singleton_components"],
                    morphology_row["singleton_components"],
                ),
            }
        )

    complexity_keys = (
        "components_8_connected",
        "singleton_components",
        "boundary_pixels_4_neighbor",
    )
    summary = {
        "grid": grid,
        "class_codes": class_codes,
        "common_valid_pixels": int(valid.sum()),
        "changed_pixels": int(changed.sum()),
        "changed_fraction": float(changed.sum() / valid.sum()),
        "control": control_summary,
        "morphology": morphology_summary,
        "relative_change_morphology_vs_control": {
            key: _relative_change(control_summary[key], morphology_summary[key])
            for key in complexity_keys
        },
        "control_map_sha256": sha256_file(control_path),
        "morphology_map_sha256": sha256_file(morphology_path),
        "changed_raster_sha256": sha256_file(changed_path),
    }
    return summary, per_class


def _point_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[column] for column in _POINT_KEY)


def _is_valid(row: dict[str, str]) -> bool:
    return row["map_valid"].strip().lower() in {"1", "true", "yes"}


def _exact_mcnemar_p(control_only: int, morphology_only: int) -> float:
    discordant = control_only + morphology_only
    if discordant == 0:
        return 1.0
    smaller = min(control_only, morphology_only)
    tail = sum(
        math.comb(discordant, index) * 0.5**discordant
        for index in range(smaller + 1)
    )
    return min(1.0, 2.0 * tail)


def _per_class_metrics(path: Path) -> dict[int, dict[str, str]]:
    rows = _read_csv(path)
    return {int(row["code"]): row for row in rows}


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _compare_points(
    control_root: Path,
    morphology_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    control_directory = control_root / "independent_evaluation"
    morphology_directory = morphology_root / "independent_evaluation"
    control_marker = _load_json(
        control_directory / "INDEPENDENT_EVALUATION_COMPLETE.json"
    )
    morphology_marker = _load_json(
        morphology_directory / "INDEPENDENT_EVALUATION_COMPLETE.json"
    )
    if control_marker.get("status") != "complete" or morphology_marker.get(
        "status"
    ) != "complete":
        raise ValueError("Both independent evaluations must be complete")
    if (
        control_marker["verified_points_sha256"]
        != morphology_marker["verified_points_sha256"]
    ):
        raise ValueError("Independent evaluations do not use the same point archive")

    control_rows = _read_csv(control_directory / "verified_point_predictions.csv")
    morphology_rows = _read_csv(
        morphology_directory / "verified_point_predictions.csv"
    )
    morphology_by_key = {_point_key(row): row for row in morphology_rows}
    if len(morphology_by_key) != len(morphology_rows):
        raise ValueError("Morphology point predictions contain duplicate point keys")

    paired_counts = {
        "both_correct": 0,
        "control_only_correct": 0,
        "morphology_only_correct": 0,
        "both_wrong": 0,
    }
    changed_predictions = 0
    per_truth: dict[int, dict[str, int]] = {}
    evaluated = 0
    for control_row in control_rows:
        morphology_row = morphology_by_key.get(_point_key(control_row))
        if morphology_row is None:
            raise ValueError(f"Point is absent from morphology evaluation: {_point_key(control_row)}")
        if not (_is_valid(control_row) and _is_valid(morphology_row)):
            continue
        evaluated += 1
        control_correct = control_row["predicted_code"] == control_row["truth_code"]
        morphology_correct = (
            morphology_row["predicted_code"] == morphology_row["truth_code"]
        )
        if control_correct and morphology_correct:
            outcome = "both_correct"
        elif control_correct:
            outcome = "control_only_correct"
        elif morphology_correct:
            outcome = "morphology_only_correct"
        else:
            outcome = "both_wrong"
        paired_counts[outcome] += 1
        changed = control_row["predicted_code"] != morphology_row["predicted_code"]
        changed_predictions += int(changed)

        truth = int(control_row["truth_code"])
        class_counts = per_truth.setdefault(
            truth,
            {
                "support": 0,
                "both_correct": 0,
                "control_only_correct": 0,
                "morphology_only_correct": 0,
                "both_wrong": 0,
                "changed_predictions": 0,
            },
        )
        class_counts["support"] += 1
        class_counts[outcome] += 1
        class_counts["changed_predictions"] += int(changed)

    if evaluated != int(control_marker["evaluated_points"]):
        raise ValueError(
            f"Paired evaluation has {evaluated} points, expected "
            f"{control_marker['evaluated_points']}"
        )

    control_metrics = _per_class_metrics(
        control_directory / "verified_per_class_metrics.csv"
    )
    morphology_metrics = _per_class_metrics(
        morphology_directory / "verified_per_class_metrics.csv"
    )
    class_rows: list[dict[str, Any]] = []
    for code in sorted(control_metrics):
        control_row = control_metrics[code]
        morphology_row = morphology_metrics[code]
        control_f1 = _optional_float(control_row.get("F1"))
        morphology_f1 = _optional_float(morphology_row.get("F1"))
        paired = per_truth.get(code, {})
        class_rows.append(
            {
                "code": code,
                "class": control_row["class"],
                "has_verified_ground_truth": control_row[
                    "has_verified_ground_truth"
                ],
                "support": int(control_row["support"]),
                "control_f1": control_f1,
                "morphology_f1": morphology_f1,
                "f1_delta_morphology_minus_control": (
                    None
                    if control_f1 is None or morphology_f1 is None
                    else morphology_f1 - control_f1
                ),
                "control_only_correct": paired.get("control_only_correct", 0),
                "morphology_only_correct": paired.get("morphology_only_correct", 0),
                "changed_predictions": paired.get("changed_predictions", 0),
            }
        )

    control_only = paired_counts["control_only_correct"]
    morphology_only = paired_counts["morphology_only_correct"]
    summary = {
        "verified_points_sha256": control_marker["verified_points_sha256"],
        "evaluated_points": evaluated,
        "control": {
            "oa": float(control_marker["OA"]),
            "macro_f1_verified_classes": float(
                control_marker["macro_F1_verified_classes"]
            ),
            "macro_recall_verified_classes": float(
                control_marker["macro_recall_verified_classes"]
            ),
        },
        "morphology": {
            "oa": float(morphology_marker["OA"]),
            "macro_f1_verified_classes": float(
                morphology_marker["macro_F1_verified_classes"]
            ),
            "macro_recall_verified_classes": float(
                morphology_marker["macro_recall_verified_classes"]
            ),
        },
        "delta_morphology_minus_control": {
            "oa": float(morphology_marker["OA"] - control_marker["OA"]),
            "macro_f1_verified_classes": float(
                morphology_marker["macro_F1_verified_classes"]
                - control_marker["macro_F1_verified_classes"]
            ),
            "macro_recall_verified_classes": float(
                morphology_marker["macro_recall_verified_classes"]
                - control_marker["macro_recall_verified_classes"]
            ),
        },
        "paired_outcomes": paired_counts,
        "changed_predictions": changed_predictions,
        "changed_prediction_fraction": changed_predictions / evaluated,
        "mcnemar_exact_two_sided_p": _exact_mcnemar_p(
            control_only,
            morphology_only,
        ),
    }
    return summary, class_rows


def _metric_delta(
    control: dict[str, Any],
    morphology: dict[str, Any],
    keys: Iterable[str],
) -> dict[str, float]:
    return {key: float(morphology[key] - control[key]) for key in keys}


def _plot_comparison(
    control_training: dict[str, Any],
    morphology_training: dict[str, Any],
    point_summary: dict[str, Any],
    point_rows: list[dict[str, Any]],
    map_summary: dict[str, Any],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for label, color, training in (
        ("DMI control", "#1f77b4", control_training),
        ("DMI + logit closing", "#ff7f0e", morphology_training),
    ):
        steps = [_int(row, "optimizer_step") for row in training["history"]]
        axes[0, 0].plot(
            steps,
            [_float(row, "validation_macro_f1_11") for row in training["history"]],
            color=color,
            marker="o",
            markersize=2.5,
            label=f"{label} macro-F1",
        )
        axes[0, 0].plot(
            steps,
            [_float(row, "validation_macro_iou_11") for row in training["history"]],
            color=color,
            linestyle="--",
            alpha=0.8,
            label=f"{label} mIoU",
        )
    axes[0, 0].set_title("Spatial validation")
    axes[0, 0].set_xlabel("DMI optimizer step")
    axes[0, 0].set_ylabel("Metric")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    metrics = ("oa", "macro_f1_verified_classes", "macro_recall_verified_classes")
    x = np.arange(len(metrics))
    width = 0.36
    axes[0, 1].bar(
        x - width / 2,
        [point_summary["control"][key] for key in metrics],
        width,
        label="DMI control",
    )
    axes[0, 1].bar(
        x + width / 2,
        [point_summary["morphology"][key] for key in metrics],
        width,
        label="DMI + logit closing",
    )
    axes[0, 1].set_xticks(x, ["OA", "Macro-F1", "Macro recall"])
    axes[0, 1].set_ylim(0.85, 0.95)
    axes[0, 1].set_title("Independent verified points")
    axes[0, 1].grid(axis="y", alpha=0.25)
    axes[0, 1].legend(fontsize=8)

    complexity = map_summary["relative_change_morphology_vs_control"]
    complexity_keys = (
        "components_8_connected",
        "singleton_components",
        "boundary_pixels_4_neighbor",
    )
    axes[1, 0].bar(
        ["Components", "Singletons", "Boundary pixels"],
        [100 * complexity[key] for key in complexity_keys],
        color="#2ca02c",
    )
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_ylabel("Relative change (%)")
    axes[1, 0].set_title("Final-map fragmentation: morphology vs control")
    axes[1, 0].grid(axis="y", alpha=0.25)

    verified_rows = [row for row in point_rows if row["support"] > 0]
    colors = [
        "#2ca02c" if row["f1_delta_morphology_minus_control"] >= 0 else "#d62728"
        for row in verified_rows
    ]
    axes[1, 1].bar(
        [f"{row['code']}:{row['class']}" for row in verified_rows],
        [row["f1_delta_morphology_minus_control"] for row in verified_rows],
        color=colors,
    )
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_ylabel("F1 delta")
    axes[1, 1].set_title("Verified-class F1: morphology minus control")
    axes[1, 1].tick_params(axis="x", rotation=35)
    axes[1, 1].grid(axis="y", alpha=0.25)

    figure.suptitle("Matched DMI comparison at LR 1e-5")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def compare_matched_runs(
    control_run_root: str | Path,
    morphology_run_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Compare completed control and morphology runs while preserving provenance."""

    control_root = Path(control_run_root).resolve()
    morphology_root = Path(morphology_run_root).resolve()
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    control_training = _history_summary(control_root)
    morphology_training = _history_summary(morphology_root)
    if control_training["end_step"] != morphology_training["end_step"]:
        raise ValueError("Matched runs have different training budgets")
    control_protocol = _matched_protocol(control_root)
    morphology_protocol = _matched_protocol(morphology_root)
    if control_protocol != morphology_protocol:
        differing = [
            key
            for key in control_protocol
            if control_protocol[key] != morphology_protocol.get(key)
        ]
        raise ValueError(
            "Runs are not a matched comparison; controlled protocol differs for: "
            + ", ".join(differing)
        )

    control_map_marker = _load_json(control_root / "map" / "MAP_COMPLETE.json")
    morphology_map_marker = _load_json(
        morphology_root / "map" / "MAP_COMPLETE.json"
    )
    control_map = control_root / "map" / control_map_marker["map"]
    morphology_map = morphology_root / "map" / morphology_map_marker["map"]
    if sha256_file(control_map) != control_map_marker["map_sha256"]:
        raise ValueError("Control map checksum does not match MAP_COMPLETE.json")
    if sha256_file(morphology_map) != morphology_map_marker["map_sha256"]:
        raise ValueError("Morphology map checksum does not match MAP_COMPLETE.json")
    if (
        control_map_marker["source_checkpoint_sha256"]
        != control_training["final_model_sha256"]
    ):
        raise ValueError("Control map was not generated from the exported final model")
    if (
        morphology_map_marker["source_checkpoint_sha256"]
        != morphology_training["final_model_sha256"]
    ):
        raise ValueError("Morphology map was not generated from the exported final model")

    map_summary, map_rows = _compare_maps(control_map, morphology_map, output_root)
    point_summary, point_rows = _compare_points(control_root, morphology_root)
    selected_keys = (
        "oa",
        "macro_f1_11",
        "macro_iou_11",
        "validation_dmi_loss",
        "fixed_train_panel_dmi_loss",
    )
    result = {
        "schema_version": "xtseg-matched-comparison-v1",
        "status": "complete",
        "controlled_factor": "depthwise_3x3_logit_additive_closing",
        "matched_protocol": control_protocol,
        "control": {
            key: value
            for key, value in control_training.items()
            if key != "history"
        },
        "morphology": {
            key: value
            for key, value in morphology_training.items()
            if key != "history"
        },
        "selected_checkpoint_delta_morphology_minus_control": _metric_delta(
            control_training["selected"],
            morphology_training["selected"],
            selected_keys,
        ),
        "independent_verified_points": point_summary,
        "final_map": map_summary,
        "scope": {
            "matched_single_seed_single_scene": True,
            "independent_points_used_for_selection": False,
            "mcnemar_tests_oa_discordance_only": True,
            "map_complexity_is_not_accuracy": True,
        },
    }

    comparison_path = output_root / "matched_comparison.json"
    map_table_path = output_root / "matched_map_per_class.csv"
    point_table_path = output_root / "matched_verified_points_per_class.csv"
    figure_path = output_root / "matched_comparison.png"
    atomic_json(result, comparison_path)
    _atomic_csv(map_rows, map_table_path)
    _atomic_csv(point_rows, point_table_path)
    _plot_comparison(
        control_training,
        morphology_training,
        point_summary,
        point_rows,
        map_summary,
        figure_path,
    )
    completion = {
        "status": "complete",
        "checkpoint": "MATCHED_CONTROL_COMPARISON",
        "schema_version": result["schema_version"],
        "control_experiment_id": control_training["experiment_id"],
        "morphology_experiment_id": morphology_training["experiment_id"],
        "artifacts": {
            path.name: sha256_file(path)
            for path in (
                comparison_path,
                map_table_path,
                point_table_path,
                figure_path,
                output_root / "matched_map_changed_pixels.tif",
            )
        },
    }
    atomic_json(completion, output_root / "MATCHED_COMPARISON_COMPLETE.json")
    return completion

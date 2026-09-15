import csv
import json

import pytest

pytest.importorskip("matplotlib")
rasterio = pytest.importorskip("rasterio")
pytest.importorskip("scipy")

import numpy as np
from rasterio.transform import from_origin

from xuanthuy_seg.comparison import compare_matched_runs
from xuanthuy_seg.contracts import sha256_file


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_run(root, experiment_id, values, classes, points):
    phase = root / "training" / "phases" / "00_dmi"
    phase.mkdir(parents=True)
    history = [
        {
            "optimizer_step": 0,
            "validation_oa": 0.5,
            "validation_macro_f1_11": 0.4,
            "validation_macro_iou_11": 0.3,
            "validation_dmi_loss": 60.0,
            "fixed_train_panel_dmi_loss": 40.0,
            "predicted_classes": 2,
        },
        {
            "optimizer_step": 10,
            "validation_oa": values[0],
            "validation_macro_f1_11": values[1],
            "validation_macro_iou_11": values[2],
            "validation_dmi_loss": values[3],
            "fixed_train_panel_dmi_loss": values[4],
            "predicted_classes": 2,
        },
    ]
    _write_csv(phase / "validation_history.csv", history)
    completion = {
        "status": "complete",
        "experiment_id": experiment_id,
        "method_hash": f"hash-{experiment_id}",
        "end_global_step": 10,
        "final_model_sha256": f"final-{experiment_id}",
        "phases": [
            {
                "best_accepted_step": 10,
                "selected_checkpoint_sha256": f"selected-{experiment_id}",
            }
        ],
    }
    (root / "training" / "TRAINING_COMPLETE.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )
    resolved = {
        "seed": 7,
        "dataset": {"dataset_id": "same-dataset"},
        "split": {"split_id": "same-split"},
        "phases": [
            {
                "loss": {"loss_id": "same-dmi"},
                "optimizer": {"type": "sgd", "learning_rate": 1e-5},
                "maximum_steps": 10,
                "validation_interval_steps": 10,
                "monitor_interval_steps": 1,
                "initialization": {"sha256": "same-ce-checkpoint"},
            }
        ],
        "selection": {"checkpoint": {"metric": "fixed_train_panel_dmi_loss"}},
        "acceptance": {"required_predicted_classes": None},
        "budget": {"batch_size": 2, "dmi_steps": 10},
    }
    (root / "training" / "resolved_config.json").write_text(
        json.dumps(resolved), encoding="utf-8"
    )

    map_directory = root / "map"
    map_directory.mkdir()
    map_path = map_directory / "prediction.tif"
    with rasterio.open(
        map_path,
        "w",
        driver="GTiff",
        height=classes.shape[0],
        width=classes.shape[1],
        count=1,
        dtype="uint8",
        nodata=0,
        crs="EPSG:32648",
        transform=from_origin(0, 4, 1, 1),
    ) as destination:
        destination.write(classes.astype("uint8"), 1)
    (map_directory / "MAP_COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "map": "prediction.tif",
                "map_sha256": sha256_file(map_path),
                "source_checkpoint_sha256": f"final-{experiment_id}",
            }
        ),
        encoding="utf-8",
    )

    evaluation = root / "independent_evaluation"
    evaluation.mkdir()
    _write_csv(evaluation / "verified_point_predictions.csv", points)
    per_class = []
    for code, name, support in ((1, "one", 2), (2, "two", 1)):
        truth = [row for row in points if int(row["truth_code"]) == code]
        correct = sum(row["truth_code"] == row["predicted_code"] for row in truth)
        predicted = sum(int(row["predicted_code"]) == code for row in points)
        precision = correct / predicted if predicted else 0.0
        recall = correct / support
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "code": code,
                "class": name,
                "has_verified_ground_truth": True,
                "support": support,
                "predicted": predicted,
                "precision": precision,
                "recall": recall,
                "F1": f1,
            }
        )
    _write_csv(evaluation / "verified_per_class_metrics.csv", per_class)
    correct = sum(row["truth_code"] == row["predicted_code"] for row in points)
    marker = {
        "status": "complete",
        "verified_points_sha256": "same-points",
        "evaluated_points": len(points),
        "OA": correct / len(points),
        "macro_F1_verified_classes": sum(row["F1"] for row in per_class) / 2,
        "macro_recall_verified_classes": sum(row["recall"] for row in per_class) / 2,
    }
    (evaluation / "INDEPENDENT_EVALUATION_COMPLETE.json").write_text(
        json.dumps(marker), encoding="utf-8"
    )


def test_compare_matched_runs_measures_fragmentation_and_paired_points(tmp_path):
    control = tmp_path / "control"
    morphology = tmp_path / "morphology"
    common = {
        "layer": "verified",
        "map_valid": True,
    }
    control_points = [
        {**common, "source_index": 0, "truth_code": 1, "predicted_code": 1, "row": 1, "col": 1},
        {**common, "source_index": 1, "truth_code": 1, "predicted_code": 2, "row": 1, "col": 2},
        {**common, "source_index": 2, "truth_code": 2, "predicted_code": 2, "row": 2, "col": 2},
    ]
    morphology_points = [
        {**common, "source_index": 0, "truth_code": 1, "predicted_code": 1, "row": 1, "col": 1},
        {**common, "source_index": 1, "truth_code": 1, "predicted_code": 1, "row": 1, "col": 2},
        {**common, "source_index": 2, "truth_code": 2, "predicted_code": 1, "row": 2, "col": 2},
    ]
    control_map = np.array(
        [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 2, 0], [0, 0, 0, 0]]
    )
    morphology_map = np.array(
        [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
    )
    _make_run(control, "control", (0.7, 0.6, 0.5, 59.0, 39.0), control_map, control_points)
    _make_run(
        morphology,
        "morphology",
        (0.71, 0.61, 0.51, 58.9, 38.9),
        morphology_map,
        morphology_points,
    )

    output = tmp_path / "comparison"
    completion = compare_matched_runs(control, morphology, output)
    result = json.loads((output / "matched_comparison.json").read_text(encoding="utf-8"))

    assert completion["status"] == "complete"
    assert result["final_map"]["common_valid_pixels"] == 4
    assert result["final_map"]["changed_pixels"] == 1
    assert result["final_map"]["control"]["components_8_connected"] == 2
    assert result["final_map"]["morphology"]["components_8_connected"] == 1
    assert result["independent_verified_points"]["paired_outcomes"] == {
        "both_correct": 1,
        "control_only_correct": 1,
        "morphology_only_correct": 1,
        "both_wrong": 0,
    }
    assert result["independent_verified_points"]["mcnemar_exact_two_sided_p"] == 1.0
    assert (output / "matched_comparison.png").is_file()
    assert (output / "matched_map_changed_pixels.tif").is_file()

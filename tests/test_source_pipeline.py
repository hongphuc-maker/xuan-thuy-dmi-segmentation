from pathlib import Path
import json
import zipfile

import numpy as np
import pytest
import yaml


torch = pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")
pd = pytest.importorskip("pandas")

from rasterio.transform import from_origin

from xuanthuy_seg.config import load_experiment_bundle, load_temporal_transfer_bundle
from xuanthuy_seg.contracts import sha256_file
from xuanthuy_seg.data.preparation import prepare_data
from xuanthuy_seg.inference import infer_full_map
from xuanthuy_seg.independent_evaluation import evaluate_verified_points
from xuanthuy_seg.temporal_transfer import (
    evaluate_temporal_transfer_points,
    infer_temporal_transfer_map,
)
from xuanthuy_seg.training.experiment import train_pipeline_experiment


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_tiny_source_first_prepare_train_and_map(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    image_path = data_root / "image.tif"
    label_path = data_root / "label.tif"
    rng = np.random.default_rng(41)
    image = rng.normal(size=(10, 64, 64)).astype(np.float32)
    label = ((np.indices((64, 64)).sum(axis=0) % 11) + 1).astype(np.uint8)
    profile = {
        "driver": "GTiff",
        "height": 64,
        "width": 64,
        "crs": "EPSG:32648",
        "transform": from_origin(400000, 2300000, 10, 10),
    }
    with rasterio.open(image_path, "w", **profile, count=10, dtype="float32") as destination:
        destination.write(image)
    with rasterio.open(label_path, "w", **profile, count=1, dtype="uint8", nodata=0) as destination:
        destination.write(label, 1)
    gpd = pytest.importorskip("geopandas")
    geometry = pytest.importorskip("shapely.geometry")
    point_source = tmp_path / "verified_source"
    point_source.mkdir()
    points = gpd.GeoDataFrame(
        {"point_id": [1, 2, 3]},
        geometry=[
            geometry.Point(400005, 2299995),
            geometry.Point(400105, 2299895),
            geometry.Point(400205, 2299795),
        ],
        crs="EPSG:32648",
    )
    points.to_file(point_source / "class-1.shp")
    point_zip = data_root / "verified.zip"
    with zipfile.ZipFile(point_zip, "w") as archive:
        for path in point_source.iterdir():
            archive.write(path, arcname=path.name)
    point_archive_bytes = point_zip.read_bytes()

    config_root = tmp_path / "configs"
    dataset = {
        "schema_version": "xtseg-dataset-v1",
        "dataset_id": "tiny",
        "expected_crs": "EPSG:32648",
        "expected_shape": [64, 64],
        "files": [
            {"role": "image", "path": "image.tif", "sha256": sha256_file(image_path)},
            {"role": "label", "path": "label.tif", "sha256": sha256_file(label_path)},
            {"role": "verified_points", "path": "verified.zip", "sha256": sha256_file(point_zip)},
        ],
        "bands": [f"B{index:02d}" for index in range(10)],
        "class_map": {index: f"class-{index}" for index in range(1, 12)},
        "validity": {"ignore_index": 255},
    }
    split = {
        "schema_version": "xtseg-split-v1",
        "split_id": "tiny-band",
        "strategy": "vertical_band",
        "parameters": {"validation_start": 24, "validation_stop": 40, "guard_px_each_side": 0},
        "patch_policy": {"size": 16, "minimum_train_fraction": 1.0, "validation_stride": 8, "inference_stride": 8},
        "sampling": {
            "manifest_seed": 1,
            "schedule_seed": 2,
            "n_patches": 16,
            "batch_size": 16,
            "rare_codes": [5, 6, 7],
            "rare_quota_each": 1,
            "dmi_support_floor_px": 1,
        },
    }
    model = {
        "schema_version": "xtseg-model-v1",
        "model_id": "tiny-unet",
        "architecture": "unet_bn",
        "parameters": {"in_channels": 10, "num_classes": 11, "base_channels": 2},
        "output": "logits",
    }
    loss = {
        "schema_version": "xtseg-loss-v1",
        "loss_id": "tiny-ce",
        "type": "ce_unweighted",
        "parameters": {"ignore_index": 255},
    }
    experiment = {
        "schema_version": "xtseg-experiment-v1",
        "experiment_id": "tiny-pipeline",
        "seed": 3,
        "dataset": "../datasets/tiny.yaml",
        "split": "../splits/tiny.yaml",
        "model": "../models/tiny.yaml",
        "phases": [
            {
                "phase_id": "ce",
                "loss": "../losses/tiny.yaml",
                "initialization": "seed",
                "optimizer": {"type": "sgd", "learning_rate": 1.0e-3, "momentum": 0.0},
                "maximum_steps": 1,
                "validation_interval_steps": 1,
                "monitor_interval_steps": 1,
            }
        ],
        "selection": {
            "checkpoint": {"metric": "validation_macro_f1_11", "direction": "max", "min_delta": 0.0},
            "stopping": {"rule": "maximum_steps"},
        },
        "acceptance": {"required_predicted_classes": None, "minimum_recall_by_code": {}},
    }
    _write_yaml(config_root / "datasets" / "tiny.yaml", dataset)
    _write_yaml(config_root / "splits" / "tiny.yaml", split)
    _write_yaml(config_root / "models" / "tiny.yaml", model)
    _write_yaml(config_root / "losses" / "tiny.yaml", loss)
    experiment_path = config_root / "experiments" / "tiny.yaml"
    _write_yaml(experiment_path, experiment)
    transfer_path = config_root / "transfers" / "tiny.yaml"
    _write_yaml(
        transfer_path,
        {
            "schema_version": "xtseg-temporal-transfer-v1",
            "transfer_id": "tiny-temporal-transfer",
            "source_experiment": "../experiments/tiny.yaml",
            "target_experiment": "../experiments/tiny.yaml",
            "normalization_policy": "source_training_artifact",
            "evaluation_interpretation": "Synthetic temporal-transfer test",
        },
    )

    bundle = load_experiment_bundle(experiment_path)
    transfer_bundle = load_temporal_transfer_bundle(transfer_path)
    artifacts = tmp_path / "artifacts"
    training = tmp_path / "training"
    maps = tmp_path / "map"
    preparation = prepare_data(bundle, data_root, artifacts)
    assert preparation["shared_train_validation_input_pixels"] == 0
    point_zip.unlink()  # Train/map must not require access to the sealed evaluation points.
    result = train_pipeline_experiment(bundle, data_root, artifacts, training, "cpu")
    assert result["status"] == "complete"
    ce_phase = training / "phases" / "00_ce"
    ce_history = pd.read_csv(ce_phase / "validation_history.csv")
    assert "validation_ce_loss" in ce_history.columns
    assert ce_history["validation_ce_weighted"].eq(False).all()
    assert (ce_phase / "best_validation_ce_loss_checkpoint.pt").is_file()
    assert (ce_phase / "best_validation_macro_f1_checkpoint.pt").is_file()

    dmi_loss = {
        "schema_version": "xtseg-loss-v1",
        "loss_id": "tiny-dmi",
        "type": "dmi_exact",
        "parameters": {
            "ignore_index": 255,
            "matrix_jitter": 0.0,
            "rank_rtol": 1.0e-12,
        },
    }
    _write_yaml(config_root / "losses" / "tiny-dmi.yaml", dmi_loss)
    dmi_experiment = {
        **experiment,
        "experiment_id": "tiny-dmi-validation-pipeline",
        "phases": [
            {
                "phase_id": "dmi",
                "loss": "../losses/tiny-dmi.yaml",
                "initialization": {
                    "type": "checkpoint",
                    "artifact": "tiny-ce/training/FINAL_MODEL.pt",
                    "sha256": sha256_file(training / "FINAL_MODEL.pt"),
                },
                "optimizer": {"type": "sgd", "learning_rate": 1.0e-7, "momentum": 0.0},
                "maximum_steps": 1,
                "validation_interval_steps": 1,
                "monitor_interval_steps": 1,
            }
        ],
        "selection": {
            "checkpoint": {"metric": "validation_dmi_loss", "direction": "min", "min_delta": 0.0},
            "stopping": {"rule": "maximum_steps"},
        },
    }
    dmi_experiment_path = config_root / "experiments" / "tiny-dmi.yaml"
    _write_yaml(dmi_experiment_path, dmi_experiment)
    dmi_bundle = load_experiment_bundle(dmi_experiment_path)
    dmi_training = tmp_path / "dmi-training"
    dmi_result = train_pipeline_experiment(
        dmi_bundle,
        data_root,
        artifacts,
        dmi_training,
        "cpu",
        training / "FINAL_MODEL.pt",
    )
    assert dmi_result["status"] == "complete"
    dmi_phase = dmi_training / "phases" / "00_dmi"
    dmi_history = pd.read_csv(dmi_phase / "validation_history.csv")
    assert {
        "validation_dmi_loss",
        "validation_dmi_log_abs_det",
        "validation_dmi_abs_det",
        "validation_dmi_rank",
        "fixed_train_panel_dmi_loss",
        "validation_macro_f1_11",
    }.issubset(dmi_history.columns)
    assert (dmi_history["validation_dmi_rank"] == 11).all()
    assert np.allclose(
        dmi_history["validation_dmi_log_abs_det"],
        -dmi_history["validation_dmi_loss"],
    )
    assert (dmi_phase / "best_validation_dmi_loss_checkpoint.pt").is_file()
    assert (dmi_phase / "best_validation_macro_f1_checkpoint.pt").is_file()
    dmi_completion = json.loads((dmi_phase / "PHASE_COMPLETE.json").read_text(encoding="utf-8"))
    assert set(dmi_completion["auxiliary_checkpoints"]) == {
        "validation_dmi_loss",
        "validation_macro_f1_11",
        "validation_macro_iou_11",
    }
    assert set(dmi_completion["model_checkpoints"]) == {
        "validation_dmi_loss",
        "validation_macro_f1_11",
        "validation_macro_iou_11",
    }
    assert dmi_completion["model_checkpoints"]["validation_dmi_loss"]["role"].startswith(
        "primary_"
    )
    checkpoint_index = json.loads(
        (dmi_phase / "MODEL_CHECKPOINTS.json").read_text(encoding="utf-8")
    )
    assert checkpoint_index == dmi_completion["model_checkpoints"]

    completion = infer_full_map(bundle, data_root, artifacts, training / "FINAL_MODEL.pt", maps, "cpu")
    assert completion["status"] == "complete"
    with rasterio.open(maps / "prediction.tif") as source:
        prediction = source.read(1)
    assert prediction.shape == (64, 64)
    assert set(np.unique(prediction)).issubset(set(range(12)))
    transfer_maps = tmp_path / "transfer-map"
    transfer_completion = infer_temporal_transfer_map(
        transfer_bundle,
        data_root,
        artifacts,
        training / "FINAL_MODEL.pt",
        transfer_maps,
        "cpu",
    )
    assert transfer_completion["status"] == "complete"
    assert transfer_completion["normalization_policy"] == "source_training_artifact"
    assert transfer_completion["source_method_hash"] == bundle.method_hash
    point_zip.write_bytes(point_archive_bytes)
    evaluation = evaluate_verified_points(
        bundle,
        data_root,
        maps / "prediction.tif",
        tmp_path / "evaluation",
    )
    assert evaluation["status"] == "complete"
    assert evaluation["evaluated_points"] == 3
    assert evaluation["verified_points_used"] is True
    transfer_evaluation = evaluate_temporal_transfer_points(
        transfer_bundle,
        data_root,
        transfer_maps / "prediction.tif",
        tmp_path / "transfer-evaluation",
    )
    assert transfer_evaluation["status"] == "complete"
    assert transfer_evaluation["evaluated_points"] == 3
    assert transfer_evaluation["transfer_hash"] == transfer_bundle.transfer_hash

import copy
from pathlib import Path

import pytest
import yaml

from xuanthuy_seg.config import load_experiment_bundle, validate_dataset


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "configs" / "experiments"
DATASETS = ROOT / "configs" / "datasets"


def test_all_public_experiment_configs_resolve() -> None:
    paths = sorted(EXPERIMENTS.glob("*.yaml"))
    assert [path.name for path in paths] == [
        "pipeline_e0w_vertical.yaml",
        "pipeline_e2w_control_vertical_weighted_ce_to_dmi_lr1e-5.yaml",
        "pipeline_e2w_vertical_weighted_ce_to_dmi_loss_driven.yaml",
        "pipeline_e3m_vertical_weighted_ce_to_dmi_logit_additive_closing_lr1e-5.yaml",
    ]
    for path in paths:
        bundle = load_experiment_bundle(path)
        assert len(bundle.method_hash) == 64
        assert bundle.resolved["method_hash"] == bundle.method_hash
        assert bundle.model["parameters"]["in_channels"] == 10
        assert bundle.model["parameters"]["num_classes"] == 11
        assert bundle.model["parameters"]["base_channels"] == 64


def test_weighted_ce_publication_protocol_is_locked() -> None:
    bundle = load_experiment_bundle(EXPERIMENTS / "pipeline_e0w_vertical.yaml")
    phase = bundle.resolved["phases"][0]

    assert bundle.method_hash == (
        "6e0e849b7cb556dcbf2862790fb7d2336586faf105416536056148fcffcd8e21"
    )
    assert phase["loss"]["type"] == "ce_weighted"
    assert phase["loss"]["parameters"]["weighting"]["source"] == (
        "prepared_data_train_core"
    )
    assert phase["optimizer"] == {
        "type": "sgd",
        "learning_rate": 1.0e-3,
        "momentum": 0.9,
        "weight_decay": 0.0,
    }
    assert phase["maximum_steps"] == 3120
    assert bundle.experiment["selection"]["checkpoint"]["metric"] == (
        "validation_macro_f1_11"
    )


def test_reported_ce_to_dmi_locks_checkpoint_lineage() -> None:
    bundle = load_experiment_bundle(
        EXPERIMENTS / "pipeline_e2w_vertical_weighted_ce_to_dmi_loss_driven.yaml"
    )
    phase = bundle.resolved["phases"][0]

    assert bundle.method_hash == (
        "e70d6976da069b4be7757e5d5889f8642dd5b98e7b9613a250751efb536b89c5"
    )
    assert phase["loss"]["type"] == "dmi_exact"
    assert phase["initialization"] == {
        "type": "checkpoint",
        "artifact": "E0w_vertical_seed20260910/training/FINAL_MODEL.pt",
        "sha256": "ffce28223708a37848811d753313ab7c97fe332030b5f44fbf78571cf7a12311",
    }
    assert phase["optimizer"]["learning_rate"] == 3.0e-7
    assert phase["maximum_steps"] == 6240
    assert bundle.experiment["selection"]["checkpoint"]["metric"] == (
        "fixed_panel_dmi_loss"
    )


def test_morphology_and_control_change_only_the_model() -> None:
    control = load_experiment_bundle(
        EXPERIMENTS / "pipeline_e2w_control_vertical_weighted_ce_to_dmi_lr1e-5.yaml"
    )
    morphology = load_experiment_bundle(
        EXPERIMENTS
        / "pipeline_e3m_vertical_weighted_ce_to_dmi_logit_additive_closing_lr1e-5.yaml"
    )
    control_phase = control.resolved["phases"][0]
    morphology_phase = morphology.resolved["phases"][0]

    assert control.method_hash == (
        "813f6be070dc62163f8aa31f645ea78d71db675c1124d66a2ec86b1340916188"
    )
    assert morphology.method_hash == (
        "e0d7c3d3460f6203fba81c7773144e5827451374d024795a84c60b3aeab1f373"
    )
    assert control.dataset == morphology.dataset
    assert control.split == morphology.split
    assert control_phase["loss"] == morphology_phase["loss"]
    assert control_phase["optimizer"] == morphology_phase["optimizer"]
    assert control_phase["maximum_steps"] == morphology_phase["maximum_steps"] == 6240
    assert control.experiment["selection"] == morphology.experiment["selection"]
    assert control.experiment["budget"] == morphology.experiment["budget"]
    assert control.model["architecture"] == "unet_bn"
    assert "factory" not in control.model
    assert morphology.model["factory"].endswith(
        ":build_unet_with_logit_additive_closing"
    )
    assert "target_module" not in control_phase["initialization"]
    assert morphology_phase["initialization"]["target_module"] == "unet"


def test_spatial_protocol_prevents_shared_input_pixels() -> None:
    bundle = load_experiment_bundle(EXPERIMENTS / "pipeline_e0w_vertical.yaml")
    split = bundle.split

    assert split["strategy"] == "vertical_band"
    assert split["parameters"] == {
        "validation_start": 1122,
        "validation_stop": 1389,
        "guard_px_each_side": 48,
    }
    assert split["patch_policy"]["size"] == 224
    assert split["sampling"]["n_patches"] == 4992
    assert split["sampling"]["batch_size"] == 16
    assert split["audit"]["require_zero_shared_input_pixels"] is True


def test_canonical_dataset_records_unknown_historical_label_time() -> None:
    dataset = yaml.safe_load(
        (
            DATASETS / "xuanthuy_may2026_historical_labels_unknown_date.yaml"
        ).read_text(encoding="utf-8")
    )
    validate_dataset(dataset)

    assert dataset["schema_version"] == "xtseg-dataset-v2"
    assert dataset["dataset_id"] == (
        "xuanthuy_s2_2026-05-26_historical_labels_unknown_date"
    )
    label = next(item for item in dataset["files"] if item["role"] == "label")
    assert label["observation_time_status"] == "unknown"
    assert "reference_date" not in label
    assert "date_precision" not in label

    invalid = copy.deepcopy(dataset)
    invalid_label = next(
        item for item in invalid["files"] if item["role"] == "label"
    )
    invalid_label["reference_date"] = "2026-01-01"
    with pytest.raises(ValueError, match="unknown label observation time"):
        validate_dataset(invalid)

from pathlib import Path

import yaml

from xuanthuy_seg import __version__
from xuanthuy_seg.config import load_experiment_bundle


ROOT = Path(__file__).resolve().parents[1]


def test_release_metadata_is_consistent() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    runtime = yaml.safe_load((ROOT / "paper" / "runtime.yaml").read_text(encoding="utf-8"))

    assert __version__ == "0.1.0"
    assert 'version = "0.1.0"' in pyproject
    assert citation["version"] == __version__
    assert runtime["common"]["seed"] == 20260910
    assert runtime["common"]["deterministic_warn_only"] is False


def test_publication_manifest_matches_resolved_configs() -> None:
    manifest = yaml.safe_load(
        (ROOT / "paper" / "experiments.yaml").read_text(encoding="utf-8")
    )
    results = yaml.safe_load(
        (ROOT / "paper" / "results.yaml").read_text(encoding="utf-8")
    )["results"]

    keys = []
    for experiment in manifest["experiments"]:
        keys.append(experiment["key"])
        bundle = load_experiment_bundle(ROOT / experiment["config"])
        assert bundle.experiment["experiment_id"] == experiment["experiment_id"]
        assert bundle.method_hash == experiment["method_hash"]
    assert set(keys) == set(results)

    control = next(
        item for item in manifest["experiments"] if item["key"] == "ce_to_dmi_lr1e_5_control"
    )
    control_results = results["ce_to_dmi_lr1e_5_control"]

    assert control["execution_status"] == "completed"
    assert control["selected_step"] == 6084
    assert len(control["selected_checkpoint_sha256"]) == 64
    assert len(control["exported_final_model_sha256"]) == 64
    assert control_results["execution_status"] == "completed"
    assert control_results["spatial_validation"]["primary_selected_step"] == 6084
    assert control_results["independent_verified_points"] is not None
    assert control_results["independent_verified_points"]["evaluated_points"] == 1036
    assert len(control_results["independent_verified_points"]["map_sha256"]) == 64
    comparison = yaml.safe_load(
        (ROOT / "paper" / "results.yaml").read_text(encoding="utf-8")
    )["comparisons"]["matched_lr1e_5_morphology"]
    assert comparison["status"] == "completed"
    assert comparison["final_map_structure"]["changed_pixels"] == 9988


def test_dataset_contract_uses_relative_paths_and_checksums() -> None:
    dataset = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "datasets"
            / "xuanthuy_may2026_labels_jan2026.yaml"
        ).read_text(encoding="utf-8")
    )
    for item in dataset["files"]:
        path = Path(item["path"])
        assert not path.is_absolute()
        assert len(item["sha256"]) == 64

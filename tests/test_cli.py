import pytest

from xuanthuy_seg.cli import build_parser, command_evaluate_points


def test_pipeline_defaults_to_train_and_does_not_open_independent_points() -> None:
    arguments = build_parser().parse_args(
        [
            "run-pipeline",
            "--experiment", "experiment.yaml",
            "--data-root", "data",
            "--run-root", "run",
        ]
    )
    assert arguments.through == "train"
    assert not arguments.confirm_independent_evaluation


def test_verify_data_accepts_training_only_roles() -> None:
    arguments = build_parser().parse_args(
        [
            "verify-data",
            "--experiment", "experiment.yaml",
            "--data-root", "data",
            "--roles", "image", "label",
        ]
    )
    assert arguments.roles == ["image", "label"]


def test_point_evaluation_requires_explicit_confirmation_before_loading_data() -> None:
    arguments = build_parser().parse_args(
        [
            "evaluate-points",
            "--experiment", "missing.yaml",
            "--data-root", "data",
            "--map", "prediction.tif",
            "--output", "evaluation",
        ]
    )
    with pytest.raises(ValueError, match="sealed"):
        command_evaluate_points(arguments)


def test_temporal_transfer_cli_keeps_source_artifacts_separate_from_target_data() -> None:
    arguments = build_parser().parse_args(
        [
            "infer-temporal-transfer",
            "--transfer", "transfer.yaml",
            "--data-root", "target-data",
            "--source-artifact-root", "source-artifacts",
            "--checkpoint", "source-model.pt",
            "--output", "transfer-map",
        ]
    )
    assert arguments.data_root == "target-data"
    assert arguments.source_artifact_root == "source-artifacts"
    assert arguments.checkpoint == "source-model.pt"


def test_plot_training_history_accepts_two_run_roots() -> None:
    arguments = build_parser().parse_args(
        [
            "plot-training-history",
            "--ce-run-root", "ce-run",
            "--dmi-run-root", "dmi-run",
            "--output", "history.png",
        ]
    )
    assert arguments.ce_run_root == "ce-run"
    assert arguments.dmi_run_root == "dmi-run"
    assert arguments.rolling_steps == 100

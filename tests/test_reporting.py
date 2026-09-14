import json

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

from xuanthuy_seg.reporting import plot_ce_to_dmi_history


def _write_phase(root, phase_name, history, steps, loss_type=None):
    phase = root / "training" / "phases" / phase_name
    phase.mkdir(parents=True)
    pd.DataFrame(history).to_csv(phase / "validation_history.csv", index=False)
    pd.DataFrame(steps).to_csv(phase / "training_step_log.csv", index=False)
    if loss_type:
        (phase / "resolved_loss_runtime.json").write_text(
            json.dumps({"type": loss_type, "loss_id": f"test-{loss_type}"}),
            encoding="utf-8",
        )


def test_plot_ce_to_dmi_history_uses_lineage_transition_and_validation_dmi(tmp_path) -> None:
    ce_root = tmp_path / "ce"
    dmi_root = tmp_path / "dmi"
    _write_phase(
        ce_root,
        "00_ce",
        [
            {"optimizer_step": 0, "validation_oa": 0.2, "validation_macro_f1_11": 0.1,
             "validation_macro_iou_11": 0.05, "validation_ce_loss": 1.4},
            {"optimizer_step": 20, "validation_oa": 0.8, "validation_macro_f1_11": 0.6,
             "validation_macro_iou_11": 0.5, "validation_ce_loss": 0.4},
            {"optimizer_step": 30, "validation_oa": 0.7, "validation_macro_f1_11": 0.5,
             "validation_macro_iou_11": 0.4, "validation_ce_loss": 0.5},
        ],
        [{"global_step": step, "loss": 1.0 / step} for step in range(1, 31)],
    )
    _write_phase(
        dmi_root,
        "00_dmi",
        [
            {"phase_step": 0, "global_step": 0, "validation_oa": 0.8,
             "validation_macro_f1_11": 0.6, "validation_macro_iou_11": 0.5,
             "validation_dmi_loss": 40.0, "fixed_train_panel_dmi_loss": 39.0},
            {"phase_step": 10, "global_step": 10, "validation_oa": 0.82,
             "validation_macro_f1_11": 0.62, "validation_macro_iou_11": 0.52,
             "validation_dmi_loss": 39.8, "fixed_train_panel_dmi_loss": 38.9},
        ],
        [{"phase_step": step, "global_step": step, "loss": 40.0 - step / 100} for step in range(1, 11)],
    )
    (dmi_root / "training" / "resolved_config.json").write_text(
        json.dumps({"lineage": {"pretrained_selected_step": 20}}),
        encoding="utf-8",
    )
    output = tmp_path / "report" / "history.png"
    result = plot_ce_to_dmi_history(ce_root, dmi_root, output, rolling_steps=3)
    assert result["status"] == "complete"
    assert result["transition_step"] == 20
    assert result["validation_ce_available"] is True
    assert result["validation_dmi_available"] is True
    assert output.is_file()
    assert len(result["sha256"]) == 64


def test_plot_discovers_ce_and_dmi_inside_one_two_phase_run(tmp_path) -> None:
    run_root = tmp_path / "combined"
    _write_phase(
        run_root,
        "00_ce",
        [{"phase_step": 0, "global_step": 0, "validation_oa": 0.2,
          "validation_macro_f1_11": 0.1, "validation_macro_iou_11": 0.05,
          "validation_ce_loss": 1.0}],
        [{"phase_step": 1, "global_step": 1, "loss": 0.9}],
        "ce_weighted",
    )
    _write_phase(
        run_root,
        "01_dmi",
        [{"phase_step": 0, "global_step": 20, "validation_oa": 0.3,
          "validation_macro_f1_11": 0.2, "validation_macro_iou_11": 0.1,
          "validation_dmi_loss": 40.0, "fixed_train_panel_dmi_loss": 39.0}],
        [{"phase_step": 1, "global_step": 21, "loss": 39.9}],
        "dmi_exact",
    )
    output = tmp_path / "combined.png"
    result = plot_ce_to_dmi_history(run_root, run_root, output)
    assert result["transition_step"] == 20
    assert result["ce_phase"] == "00_ce"
    assert result["dmi_phase"] == "01_dmi"
    assert output.is_file()

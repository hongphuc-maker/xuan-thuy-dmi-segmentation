from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import sha256_file


def _training_root(path: str | Path) -> Path:
    root = Path(path).resolve()
    if (root / "training" / "phases").is_dir():
        return root / "training"
    if (root / "phases").is_dir():
        return root
    raise FileNotFoundError(f"Cannot find training/phases under {root}")


def _phase_for_objective(training_root: Path, objective: str) -> Path:
    phases = sorted(
        path for path in (training_root / "phases").iterdir()
        if (path / "validation_history.csv").is_file()
    )
    if len(phases) == 1:
        return phases[0]
    matches: list[Path] = []
    for phase in phases:
        loss_path = phase / "resolved_loss_runtime.json"
        if not loss_path.is_file():
            continue
        loss = json.loads(loss_path.read_text(encoding="utf-8"))
        identity = f"{loss.get('type', '')} {loss.get('loss_id', '')}".lower()
        if objective in identity:
            matches.append(phase)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {objective.upper()} phase under {training_root}; "
            f"found {len(matches)} matches among {len(phases)} phases"
        )
    return matches[0]


def _step_column(frame: pd.DataFrame) -> str:
    for column in ("global_step", "optimizer_step", "phase_step"):
        if column in frame.columns:
            return column
    raise ValueError("History has no global_step, optimizer_step or phase_step column")


def _infer_transition_step(dmi_training: Path, ce_history: pd.DataFrame) -> int:
    config_path = dmi_training / "resolved_config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        selected = config.get("lineage", {}).get("pretrained_selected_step")
        if selected is not None:
            return int(selected)
    step_column = _step_column(ce_history)
    return int(ce_history[step_column].max())


def _local_steps(frame: pd.DataFrame) -> pd.Series:
    if "phase_step" in frame.columns:
        return frame["phase_step"].astype(int)
    step_column = _step_column(frame)
    return frame[step_column].astype(int) - int(frame[step_column].min())


def plot_ce_to_dmi_history(
    ce_run_root: str | Path,
    dmi_run_root: str | Path,
    output: str | Path,
    transition_step: int | None = None,
    rolling_steps: int = 100,
) -> dict[str, Any]:
    """Plot a truthful CE→DMI trajectory from versioned training artifacts.

    CE and DMI objective values use separate y axes because their numerical
    scales and definitions differ. The DMI validation curve is available for
    runs produced by the validation-DMI protocol revision.
    """
    import matplotlib.pyplot as plt

    ce_training = _training_root(ce_run_root)
    dmi_training = _training_root(dmi_run_root)
    ce_phase = _phase_for_objective(ce_training, "ce")
    dmi_phase = _phase_for_objective(dmi_training, "dmi")
    ce_history = pd.read_csv(ce_phase / "validation_history.csv")
    dmi_history = pd.read_csv(dmi_phase / "validation_history.csv")
    ce_steps = pd.read_csv(ce_phase / "training_step_log.csv")
    dmi_steps = pd.read_csv(dmi_phase / "training_step_log.csv")
    if transition_step is not None:
        transition = int(transition_step)
    elif ce_training == dmi_training and "global_step" in dmi_history:
        transition = int(dmi_history["global_step"].min())
    else:
        transition = _infer_transition_step(dmi_training, ce_history)

    ce_history_x = ce_history[_step_column(ce_history)].astype(int)
    ce_steps_x = ce_steps[_step_column(ce_steps)].astype(int)
    ce_history_keep = ce_history_x <= transition
    ce_steps_keep = ce_steps_x <= transition
    dmi_history_x = transition + _local_steps(dmi_history)
    dmi_steps_x = transition + _local_steps(dmi_steps)

    figure, (metric_axis, loss_axis) = plt.subplots(1, 2, figsize=(16, 5.5))
    metric_specs = (
        ("validation_oa", "OA", "#1f77b4"),
        ("validation_macro_f1_11", "macro-F1 (11 classes)", "#2ca02c"),
        ("validation_macro_iou_11", "macro-IoU (11 classes)", "#9467bd"),
    )
    for column, label, color in metric_specs:
        if column in ce_history:
            metric_axis.plot(
                ce_history_x[ce_history_keep],
                ce_history.loc[ce_history_keep, column],
                color=color,
                label=f"CE — {label}",
            )
        if column in dmi_history:
            metric_axis.plot(
                dmi_history_x,
                dmi_history[column],
                color=color,
                linestyle="--",
                marker="o",
                markersize=3,
                label=f"DMI — {label}",
            )
    metric_axis.axvline(transition, color="black", linestyle=":", label="CE → DMI")
    metric_axis.set_title("Spatial-validation performance")
    metric_axis.set_xlabel("Combined optimizer step")
    metric_axis.set_ylabel("Metric")
    metric_axis.set_ylim(0.0, 1.0)
    metric_axis.grid(alpha=0.25)
    metric_axis.legend(fontsize=8, ncol=2)

    window = max(1, int(rolling_steps))
    ce_loss = ce_steps["loss"].rolling(window, min_periods=1).mean()
    loss_axis.plot(
        ce_steps_x[ce_steps_keep],
        ce_loss[ce_steps_keep],
        color="#1f77b4",
        label=f"CE train loss — rolling {window}",
    )
    if "validation_ce_loss" in ce_history:
        loss_axis.plot(
            ce_history_x[ce_history_keep],
            ce_history.loc[ce_history_keep, "validation_ce_loss"],
            color="#17becf",
            marker="s",
            markersize=3,
            label="CE validation loss",
        )
    loss_axis.set_xlabel("Combined optimizer step")
    loss_axis.set_ylabel("CE loss", color="#1f77b4")
    loss_axis.tick_params(axis="y", labelcolor="#1f77b4")
    loss_axis.grid(alpha=0.25)
    loss_axis.axvline(transition, color="black", linestyle=":")

    dmi_axis = loss_axis.twinx()
    if "validation_dmi_loss" in ce_history:
        dmi_axis.plot(
            ce_history_x[ce_history_keep],
            ce_history.loc[ce_history_keep, "validation_dmi_loss"],
            color="#8c564b",
            linestyle=":",
            marker="s",
            markersize=3,
            label="Validation DMI loss during CE",
        )
    dmi_loss = dmi_steps["loss"].rolling(window, min_periods=1).mean()
    dmi_axis.plot(
        dmi_steps_x,
        dmi_loss,
        color="#ff7f0e",
        alpha=0.7,
        label=f"DMI train loss — rolling {window}",
    )
    if "fixed_train_panel_dmi_loss" in dmi_history:
        dmi_axis.plot(
            dmi_history_x,
            dmi_history["fixed_train_panel_dmi_loss"],
            color="#d62728",
            marker="o",
            markersize=3,
            label="DMI fixed train panel",
        )
    elif "fixed_panel_dmi_loss" in dmi_history:
        dmi_axis.plot(
            dmi_history_x,
            dmi_history["fixed_panel_dmi_loss"],
            color="#d62728",
            marker="o",
            markersize=3,
            label="DMI fixed train panel (legacy)",
        )
    if "validation_dmi_loss" in dmi_history:
        dmi_axis.plot(
            dmi_history_x,
            dmi_history["validation_dmi_loss"],
            color="#8c564b",
            marker="s",
            markersize=3,
            label="DMI validation loss",
        )
    dmi_axis.set_ylabel("DMI loss", color="#ff7f0e")
    dmi_axis.tick_params(axis="y", labelcolor="#ff7f0e")
    loss_axis.set_title("Phase objectives on separate loss scales")
    handles_left, labels_left = loss_axis.get_legend_handles_labels()
    handles_right, labels_right = dmi_axis.get_legend_handles_labels()
    loss_axis.legend(handles_left + handles_right, labels_left + labels_right, fontsize=8)

    figure.suptitle("Weighted CE to DMI: train monitor and spatial validation", fontsize=14)
    figure.tight_layout()
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {
        "status": "complete",
        "output": str(output_path),
        "sha256": sha256_file(output_path),
        "transition_step": transition,
        "ce_phase": ce_phase.name,
        "dmi_phase": dmi_phase.name,
        "validation_ce_available": "validation_ce_loss" in ce_history,
        "validation_dmi_available": "validation_dmi_loss" in dmi_history,
    }

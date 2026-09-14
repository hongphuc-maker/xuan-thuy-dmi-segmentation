from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from ..config import ConfigBundle
from ..contracts import atomic_json, sha256_file, stable_hash, verify_files
from ..data.cp2 import CP2Data, load_cp2_data
from ..evaluation import evaluate_spatial_validation
from ..losses import build_loss
from ..models import build_model
from ..reproducibility import configure_reproducibility, git_commit
from .optimizers import build_optimizer
from .runner import (
    _atomic_csv,
    _atomic_torch_save,
    _build_schedule,
    _cpu_state_dict,
    _metrics_row,
    _torch_load_cpu,
)
from .selection import CheckpointSelector


def _build_runtime_loss(phase: dict[str, Any], data: CP2Data, device: torch.device):
    config = copy.deepcopy(phase["loss"])
    if config["type"] == "ce_weighted":
        weighting = config["parameters"]["weighting"]
        source = str(weighting.get("source", ""))
        observed = np.asarray(
            data.contract.get("class_counts_unique_train_core", []),
            dtype=np.int64,
        )
        if source == "prepared_data_train_core":
            if len(observed) != len(data.class_map):
                train_valid = data.common_valid & (data.split_mask == 1)
                observed = np.bincount(
                    data.label[train_valid].astype(np.int64),
                    minlength=len(data.class_map) + 1,
                )[1:]
            weighting["class_counts"] = observed.tolist()
        elif "class_counts" in weighting and len(observed):
            configured = np.asarray(weighting["class_counts"], dtype=np.int64)
            if not np.array_equal(configured, observed):
                raise ValueError("Weighted-CE class counts differ from prepared train-core counts")
    return build_loss(config).to(device), config


def _selection_for_phase(bundle: ConfigBundle, phase_index: int) -> dict[str, Any]:
    phase = bundle.resolved["phases"][phase_index]
    if "selection" in phase:
        return phase["selection"]
    if phase_index < len(bundle.resolved["phases"]) - 1:
        return {
            "checkpoint": {
                "metric": "validation_macro_f1_11",
                "direction": "max",
                "min_delta": 0.001,
            },
            "stopping": {"rule": "maximum_steps"},
        }
    return bundle.experiment["selection"]


def _first_initialization(
    bundle: ConfigBundle,
    model: torch.nn.Module,
    checkpoint: str | Path | None,
) -> dict[str, Any]:
    spec = bundle.resolved["phases"][0].get("initialization", "seed")
    if isinstance(spec, str) and spec in {"seed", "shared_seed"}:
        return {"type": "seed", "seed": int(bundle.experiment["seed"])}
    if not isinstance(spec, dict) or spec.get("type") != "checkpoint":
        raise ValueError("First phase initialization must be seed/shared_seed or a checkpoint mapping")
    if checkpoint is None:
        raise ValueError("This experiment requires --initialization-checkpoint")
    path = Path(checkpoint).resolve()
    expected = str(spec["sha256"])
    if sha256_file(path) != expected:
        raise ValueError("Initialization checkpoint SHA-256 mismatch")
    payload = _torch_load_cpu(path)
    state = payload.get("model_state", payload)
    target_name = str(spec.get("target_module", "model"))
    target = model
    if target_name != "model":
        for attribute in target_name.split("."):
            target = getattr(target, attribute, None)
            if not isinstance(target, torch.nn.Module):
                raise ValueError(
                    f"Initialization target_module is not a model submodule: {target_name}"
                )
    target.load_state_dict(state, strict=True)
    return {
        "type": "checkpoint",
        "path": str(path),
        "sha256": expected,
        "target_module": target_name,
    }


def _run_lock(
    bundle: ConfigBundle,
    data_root: Path,
    artifact_root: Path,
    output: Path,
    runtime: dict[str, Any],
    initialization: dict[str, Any],
) -> None:
    lock_path = output / "RUN_LOCK.json"
    repository_root = Path(__file__).resolve().parents[3]
    current_commit = git_commit(repository_root)
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
        if lock.get("method_hash") != bundle.method_hash:
            raise ValueError("Run directory belongs to a different experiment method hash")
        if stable_hash(metadata) != lock.get("run_metadata_sha256"):
            raise ValueError("run_metadata.json differs from RUN_LOCK.json")
        if metadata.get("git_commit") != current_commit:
            raise ValueError("Refusing to resume with a different source commit")
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Non-empty training output has no RUN_LOCK.json: {output}")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment_id": bundle.experiment["experiment_id"],
        "method_hash": bundle.method_hash,
        "git_commit": current_commit,
        "data_root": str(data_root),
        "artifact_root": str(artifact_root),
        "initialization": initialization,
        "runtime": runtime,
    }
    atomic_json(bundle.resolved, output / "resolved_config.json")
    atomic_json(metadata, output / "run_metadata.json")
    atomic_json(
        {
            "status": "frozen",
            "method_hash": bundle.method_hash,
            "run_metadata_sha256": stable_hash(metadata),
        },
        lock_path,
    )


@torch.inference_mode()
def _fixed_panel_loss(
    model: torch.nn.Module,
    loss_function: torch.nn.Module,
    data: CP2Data,
    device: torch.device,
    panel_size: int = 11,
) -> float:
    was_training = model.training
    model.eval()
    indices = np.rint(np.linspace(0, len(data.frozen_batches) - 1, panel_size)).astype(int)
    values: list[float] = []
    for batch_index in indices:
        batch = data.materialize_batch(int(batch_index))
        inputs = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        loss = loss_function(model(inputs), targets)
        if isinstance(loss, tuple):
            loss = loss[0]
        values.append(float(loss.detach().cpu()))
    if was_training:
        model.train()
    return float(np.mean(values))


def _phase_checkpoint(
    path: Path,
    bundle: ConfigBundle,
    phase_id: str,
    model: torch.nn.Module,
    phase_step: int,
    global_step: int,
    metrics: dict[str, Any],
    accepted: bool,
    checkpoint_role: str = "selection",
) -> None:
    _atomic_torch_save(
        {
            "method_hash": bundle.method_hash,
            "experiment_id": bundle.experiment["experiment_id"],
            "phase_id": phase_id,
            "phase_step": int(phase_step),
            "global_step": int(global_step),
            "validation": metrics,
            "acceptance_pass": bool(accepted),
            "checkpoint_role": checkpoint_role,
            "model_config": bundle.model,
            "model_state": _cpu_state_dict(model),
        },
        path,
    )


_AUXILIARY_CHECKPOINTS = {
    "validation_ce_loss": ("min", "best_validation_ce_loss_checkpoint.pt"),
    "validation_macro_f1_11": ("max", "best_validation_macro_f1_checkpoint.pt"),
    "validation_macro_iou_11": ("max", "best_validation_macro_iou_checkpoint.pt"),
    "validation_dmi_loss": ("min", "best_validation_dmi_loss_checkpoint.pt"),
}


def _finite_history_values(history: list[dict[str, Any]], metric: str) -> list[float]:
    return [
        float(row[metric])
        for row in history
        if metric in row and math.isfinite(float(row[metric]))
    ]


def _save_auxiliary_checkpoints(
    phase_dir: Path,
    bundle: ConfigBundle,
    phase_id: str,
    model: torch.nn.Module,
    phase_step: int,
    global_step: int,
    metrics: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    """Save semantic-best and DMI-validation-best independently of selection."""
    for metric, (direction, filename) in _AUXILIARY_CHECKPOINTS.items():
        if metric not in metrics:
            continue
        current = float(metrics[metric])
        if not math.isfinite(current):
            continue
        previous = _finite_history_values(history, metric)
        improved = not previous or (
            current > max(previous) if direction == "max" else current < min(previous)
        )
        if improved:
            _phase_checkpoint(
                phase_dir / filename,
                bundle,
                phase_id,
                model,
                phase_step,
                global_step,
                metrics,
                False,
                checkpoint_role=f"best_{metric}",
            )


def _auxiliary_checkpoint_manifest(
    phase_dir: Path,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for metric, (direction, filename) in _AUXILIARY_CHECKPOINTS.items():
        path = phase_dir / filename
        candidates = [
            row for row in history
            if metric in row and math.isfinite(float(row[metric]))
        ]
        if not path.is_file() or not candidates:
            continue
        best = (max if direction == "max" else min)(
            candidates,
            key=lambda row: float(row[metric]),
        )
        manifest[metric] = {
            "direction": direction,
            "value": float(best[metric]),
            "global_step": int(best["optimizer_step"]),
            "phase_step": int(best.get("phase_step", best["optimizer_step"])),
            "checkpoint": filename,
            "checkpoint_sha256": sha256_file(path),
        }
    return manifest


def _phase_resume(
    path: Path,
    bundle: ConfigBundle,
    phase_id: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    phase_step: int,
    global_step: int,
    accepted_selector: CheckpointSelector,
    unrestricted_selector: CheckpointSelector,
    histories: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]],
) -> None:
    history, per_class, steps = histories
    _atomic_torch_save(
        {
            "method_hash": bundle.method_hash,
            "phase_id": phase_id,
            "phase_step": int(phase_step),
            "global_step": int(global_step),
            "model_state": _cpu_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "accepted_selector": accepted_selector.state_dict(),
            "unrestricted_selector": unrestricted_selector.state_dict(),
            "history": history,
            "per_class": per_class,
            "steps": steps,
        },
        path,
    )


def _metric_payload(
    model: torch.nn.Module,
    loss_function: torch.nn.Module,
    selector_config: dict[str, Any],
    data: CP2Data,
    device: torch.device,
    output_type: str,
    recent_losses: list[float],
    runtime_loss_config: dict[str, Any],
    monitor_validation_dmi: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    metric = str(selector_config["checkpoint"]["metric"])
    loss_type = str(runtime_loss_config.get("type", "")).lower()
    loss_id = str(runtime_loss_config.get("loss_id", "")).lower()
    track_dmi = "dmi" in loss_type or "dmi" in loss_id
    track_ce = loss_type.startswith("ce_") or "cross_entropy" in loss_id
    parameters = runtime_loss_config.get("parameters", {})
    summary, per_class, _ = evaluate_spatial_validation(
        model,
        data,
        device,
        output_type=output_type,
        include_dmi=track_dmi or metric == "validation_dmi_loss" or monitor_validation_dmi,
        dmi_matrix_jitter=float(parameters.get("matrix_jitter", 0.0)),
        dmi_rank_rtol=float(parameters.get("rank_rtol", 1e-12)),
        include_ce=track_ce or metric == "validation_ce_loss",
        ce_class_weights=getattr(loss_function, "class_weights", None),
    )
    if track_dmi:
        panel_loss = _fixed_panel_loss(model, loss_function, data, device)
        summary["fixed_train_panel_dmi_loss"] = panel_loss
        # Legacy alias: historical YAMLs remain resolvable and reproducible.
        summary["fixed_panel_dmi_loss"] = panel_loss
    if metric == "training_loss_ema":
        if not recent_losses:
            summary[metric] = float("inf")
        else:
            summary[metric] = float(np.mean(recent_losses[-min(100, len(recent_losses)) :]))
    return summary, per_class


def train_pipeline_experiment(
    bundle: ConfigBundle,
    data_root: str | Path,
    artifact_root: str | Path,
    output: str | Path,
    device_name: str = "cuda",
    initialization_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    data_root = Path(data_root).resolve()
    artifact_root = Path(artifact_root).resolve()
    output = Path(output).resolve()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is available")
    runtime = configure_reproducibility(int(bundle.experiment["seed"]), warn_only=False)
    verify_files(
        data_root,
        [item for item in bundle.dataset["files"] if str(item["role"]) in {"image", "label"}],
    )
    data = load_cp2_data(bundle.dataset, bundle.split, data_root, artifact_root)
    model = build_model(bundle.model).to(device)
    initialization = _first_initialization(bundle, model, initialization_checkpoint)
    _run_lock(bundle, data_root, artifact_root, output, runtime, initialization)
    root_completion = output / "TRAINING_COMPLETE.json"
    if root_completion.exists():
        result = json.loads(root_completion.read_text(encoding="utf-8"))
        if result.get("method_hash") != bundle.method_hash:
            raise ValueError("Training completion method hash mismatch")
        final_path = output / str(result.get("final_model", "FINAL_MODEL.pt"))
        if not final_path.is_file() or sha256_file(final_path) != result.get("final_model_sha256"):
            raise ValueError("Final model differs from TRAINING_COMPLETE.json")
        print(f"Training already complete: {root_completion}")
        return result

    phases = bundle.resolved["phases"]
    total_steps = sum(int(phase["maximum_steps"]) for phase in phases)
    schedule = _build_schedule(total_steps, len(data.frozen_batches), int(bundle.experiment["seed"]))
    global_step = 0
    previous_selected: Path | None = None
    phase_results: list[dict[str, Any]] = []
    for phase_index, phase in enumerate(phases):
        phase_id = str(phase["phase_id"])
        phase_dir = output / "phases" / f"{phase_index:02d}_{phase_id}"
        phase_dir.mkdir(parents=True, exist_ok=True)
        completion_path = phase_dir / "PHASE_COMPLETE.json"
        if completion_path.exists():
            phase_result = json.loads(completion_path.read_text(encoding="utf-8"))
            if phase_result.get("method_hash") != bundle.method_hash:
                raise ValueError(f"Completed phase {phase_id} has a different method hash")
            previous_selected = phase_dir / str(phase_result["selected_checkpoint"])
            if (
                not previous_selected.is_file()
                or sha256_file(previous_selected) != phase_result.get("selected_checkpoint_sha256")
            ):
                raise ValueError(f"Selected checkpoint for completed phase {phase_id} failed verification")
            payload = _torch_load_cpu(previous_selected)
            model.load_state_dict(payload["model_state"], strict=True)
            global_step = int(phase_result["end_global_step"])
            phase_results.append(phase_result)
            continue

        if phase_index > 0:
            initialization_rule = phase.get("initialization", "previous_phase_best")
            if initialization_rule not in {"previous_phase_best", "previous_phase_selected"}:
                raise ValueError(f"Unsupported phase initialization rule: {initialization_rule}")
            if previous_selected is None:
                raise ValueError("Previous phase selected checkpoint is unavailable")
            payload = _torch_load_cpu(previous_selected)
            model.load_state_dict(payload["model_state"], strict=True)

        loss_function, runtime_loss_config = _build_runtime_loss(phase, data, device)
        atomic_json(runtime_loss_config, phase_dir / "resolved_loss_runtime.json")
        optimizer = build_optimizer(model.parameters(), phase["optimizer"])
        maximum_steps = int(phase["maximum_steps"])
        validation_interval = int(phase.get("validation_interval_steps", 156))
        monitor_interval = int(phase.get("monitor_interval_steps", 10))
        selector_config = _selection_for_phase(bundle, phase_index)
        default_acceptance = (
            bundle.experiment["acceptance"]
            if phase_index == len(phases) - 1
            else {}
        )
        acceptance = phase.get("acceptance", default_acceptance)
        accepted_selector = CheckpointSelector(selector_config, acceptance)
        unrestricted_selector = CheckpointSelector(selector_config, {})
        resume_path = phase_dir / "latest_resume_checkpoint.pt"
        history: list[dict[str, Any]] = []
        per_class_rows: list[dict[str, Any]] = []
        step_rows: list[dict[str, Any]] = []
        phase_step = 0
        if resume_path.exists():
            resume = _torch_load_cpu(resume_path)
            if resume["method_hash"] != bundle.method_hash or resume["phase_id"] != phase_id:
                raise ValueError("Phase resume checkpoint contract mismatch")
            model.load_state_dict(resume["model_state"], strict=True)
            optimizer.load_state_dict(resume["optimizer_state"])
            phase_step, global_step = int(resume["phase_step"]), int(resume["global_step"])
            accepted_selector.load_state_dict(resume["accepted_selector"])
            unrestricted_selector.load_state_dict(resume["unrestricted_selector"])
            history = list(resume["history"])
            per_class_rows = list(resume["per_class"])
            step_rows = list(resume["steps"])
            print(f"AUTO-RESUME {phase_id}: {phase_step}/{maximum_steps}")

        recent_losses = [float(row["loss"]) for row in step_rows[-100:]]
        if phase_step == 0 and not history:
            metrics, per_class = _metric_payload(
                model, loss_function, selector_config, data, device,
                str(bundle.model.get("output", "logits")), recent_losses,
                runtime_loss_config,
                bool(phase.get("monitor_validation_dmi", False)),
            )
            _save_auxiliary_checkpoints(
                phase_dir, bundle, phase_id, model, 0, global_step,
                metrics, history,
            )
            unrestricted = unrestricted_selector.update(global_step, metrics)
            accepted = accepted_selector.update(global_step, metrics)
            history.append(_metrics_row(global_step, metrics, phase_step=0, accepted=accepted.eligible))
            per_class.insert(0, "global_step", global_step)
            per_class.insert(0, "phase_step", 0)
            per_class_rows.extend(json.loads(per_class.to_json(orient="records")))
            if unrestricted.improved:
                _phase_checkpoint(
                    phase_dir / "best_unrestricted_checkpoint.pt", bundle, phase_id,
                    model, 0, global_step, metrics, False,
                )
            if accepted.improved:
                _phase_checkpoint(
                    phase_dir / "best_accepted_checkpoint.pt", bundle, phase_id,
                    model, 0, global_step, metrics, True,
                )

        progress = tqdm(
            total=maximum_steps,
            initial=phase_step,
            desc=f"{bundle.experiment['experiment_id']}:{phase_id}",
            unit="step",
            dynamic_ncols=True,
        )
        early_stopped = False
        interrupted = False
        try:
            while phase_step < maximum_steps:
                schedule_row = schedule[global_step]
                batch = data.materialize_batch(schedule_row["batch_index"])
                inputs = batch["image"].to(device, non_blocking=True)
                targets = batch["target"].to(device, non_blocking=True)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                if isinstance(loss, tuple):
                    loss = loss[0]
                if not bool(torch.isfinite(loss).detach().cpu()):
                    raise FloatingPointError(f"Non-finite loss at {phase_id} step {phase_step + 1}")
                loss.backward()
                nonfinite = sum(
                    int((~torch.isfinite(parameter.grad)).sum().detach().cpu())
                    for parameter in model.parameters() if parameter.grad is not None
                )
                if nonfinite:
                    raise FloatingPointError(f"Non-finite gradients at {phase_id} step {phase_step + 1}")
                optimizer.step()
                phase_step += 1
                global_step += 1
                loss_value = float(loss.detach().cpu())
                recent_losses.append(loss_value)
                step_rows.append(
                    {
                        "phase_id": phase_id,
                        "phase_step": phase_step,
                        "global_step": global_step,
                        **schedule_row,
                        "loss": loss_value,
                        "nonfinite_gradient_values": nonfinite,
                    }
                )
                progress.update(1)
                if phase_step % monitor_interval == 0:
                    postfix = {"loss10": f"{np.mean(recent_losses[-monitor_interval:]):.4f}"}
                    if unrestricted_selector.best_value is not None:
                        postfix["best"] = f"{unrestricted_selector.best_value:.4f}"
                    if device.type == "cuda":
                        postfix["GPU"] = f"{torch.cuda.memory_allocated(device) / 2**30:.2f}G"
                        postfix["peak"] = f"{torch.cuda.max_memory_allocated(device) / 2**30:.2f}G"
                    progress.set_postfix(postfix, refresh=True)
                if phase_step % validation_interval != 0 and phase_step != maximum_steps:
                    continue
                metrics, per_class = _metric_payload(
                    model, loss_function, selector_config, data, device,
                    str(bundle.model.get("output", "logits")), recent_losses,
                    runtime_loss_config,
                    bool(phase.get("monitor_validation_dmi", False)),
                )
                _save_auxiliary_checkpoints(
                    phase_dir, bundle, phase_id, model, phase_step, global_step,
                    metrics, history,
                )
                unrestricted = unrestricted_selector.update(global_step, metrics)
                accepted = accepted_selector.update(global_step, metrics)
                history.append(
                    _metrics_row(
                        global_step, metrics, phase_step=phase_step,
                        accepted=accepted.eligible, acceptance_reason=accepted.reason,
                        improved_accepted=accepted.improved,
                        improved_unrestricted=unrestricted.improved,
                        no_improve=accepted.no_improve_evaluations,
                    )
                )
                per_class.insert(0, "global_step", global_step)
                per_class.insert(0, "phase_step", phase_step)
                per_class_rows.extend(json.loads(per_class.to_json(orient="records")))
                if unrestricted.improved:
                    _phase_checkpoint(
                        phase_dir / "best_unrestricted_checkpoint.pt", bundle, phase_id,
                        model, phase_step, global_step, metrics, False,
                    )
                if accepted.improved:
                    _phase_checkpoint(
                        phase_dir / "best_accepted_checkpoint.pt", bundle, phase_id,
                        model, phase_step, global_step, metrics, True,
                    )
                _atomic_csv(pd.DataFrame(history), phase_dir / "validation_history.csv")
                _atomic_csv(pd.DataFrame(per_class_rows), phase_dir / "validation_per_class.csv")
                _atomic_csv(pd.DataFrame(step_rows), phase_dir / "training_step_log.csv")
                _phase_resume(
                    resume_path, bundle, phase_id, model, optimizer, phase_step, global_step,
                    accepted_selector, unrestricted_selector, (history, per_class_rows, step_rows),
                )
                dmi_monitor = ""
                if "validation_dmi_loss" in metrics:
                    dmi_monitor = (
                        f" val-DMI={metrics['validation_dmi_loss']:.4f}"
                        f" rank={int(metrics['validation_dmi_rank'])}/11"
                    )
                    if "fixed_train_panel_dmi_loss" in metrics:
                        dmi_monitor += (
                            f" train-panel-DMI={metrics['fixed_train_panel_dmi_loss']:.4f}"
                        )
                ce_monitor = ""
                if "validation_ce_loss" in metrics:
                    ce_monitor = f" val-CE={metrics['validation_ce_loss']:.4f}"
                progress.write(
                    f"{phase_id} {phase_step}/{maximum_steps} global={global_step} "
                    f"OA={metrics['validation_oa']:.4f} "
                    f"macro-F1={metrics['validation_macro_f1_11']:.4f} "
                    f"predicted={metrics['predicted_classes']}/11 "
                    f"selected-metric={selector_config['checkpoint']['metric']}="
                    f"{float(metrics[selector_config['checkpoint']['metric']]):.4f} "
                    f"accepted={accepted.eligible} patience={accepted.no_improve_evaluations}/"
                    f"{accepted_selector.patience}{ce_monitor}{dmi_monitor}"
                )
                if accepted.should_stop:
                    early_stopped = True
                    break
        except KeyboardInterrupt:
            interrupted = True
            progress.write("Interrupted; saving resumable state...")
        finally:
            progress.close()
            _atomic_csv(pd.DataFrame(history), phase_dir / "validation_history.csv")
            _atomic_csv(pd.DataFrame(per_class_rows), phase_dir / "validation_per_class.csv")
            _atomic_csv(pd.DataFrame(step_rows), phase_dir / "training_step_log.csv")
            _phase_resume(
                resume_path, bundle, phase_id, model, optimizer, phase_step, global_step,
                accepted_selector, unrestricted_selector, (history, per_class_rows, step_rows),
            )
        if interrupted:
            return {
                "status": "interrupted_resume_available",
                "phase_id": phase_id,
                "phase_step": phase_step,
                "global_step": global_step,
                "resume_checkpoint": str(resume_path),
            }

        accepted_path = phase_dir / "best_accepted_checkpoint.pt"
        unrestricted_path = phase_dir / "best_unrestricted_checkpoint.pt"
        best_selected = accepted_path if accepted_path.exists() else unrestricted_path
        handoff_policy = str(phase.get("handoff_checkpoint", "selected"))
        selected = resume_path if handoff_policy == "latest" else best_selected
        auxiliary_checkpoints = _auxiliary_checkpoint_manifest(phase_dir, history)
        selection_metric = str(selector_config["checkpoint"]["metric"])
        selected_selector = (
            accepted_selector if accepted_path.exists() else unrestricted_selector
        )
        selected_history_row = next(
            row
            for row in history
            if int(row["optimizer_step"]) == int(selected_selector.best_step)
        )
        model_checkpoints = {
            selection_metric: {
                "direction": str(selector_config["checkpoint"]["direction"]),
                "value": float(selected_selector.best_value),
                "global_step": int(selected_selector.best_step),
                "phase_step": int(
                    selected_history_row.get("phase_step", selected_selector.best_step)
                ),
                "checkpoint": best_selected.name,
                "checkpoint_sha256": sha256_file(best_selected),
                "role": (
                    "primary_accepted_selection"
                    if accepted_path.exists()
                    else "primary_unrestricted_selection"
                ),
            }
        }
        for metric in ("validation_macro_f1_11", "validation_macro_iou_11"):
            if metric in auxiliary_checkpoints:
                model_checkpoints[metric] = {
                    **auxiliary_checkpoints[metric],
                    "role": "auxiliary_semantic_selection",
                }
        atomic_json(model_checkpoints, phase_dir / "MODEL_CHECKPOINTS.json")
        phase_result = {
            "status": "complete",
            "method_hash": bundle.method_hash,
            "phase_id": phase_id,
            "execution_status": "early_stopped" if early_stopped else "maximum_steps_reached",
            "end_phase_step": phase_step,
            "end_global_step": global_step,
            "acceptance_pass": accepted_path.exists(),
            "best_accepted_step": accepted_selector.best_step,
            "best_unrestricted_step": unrestricted_selector.best_step,
            "handoff_checkpoint_policy": handoff_policy,
            "best_selected_checkpoint": best_selected.name,
            "best_selected_checkpoint_sha256": sha256_file(best_selected),
            "selected_checkpoint": selected.name,
            "selected_checkpoint_sha256": sha256_file(selected),
            "auxiliary_checkpoints": auxiliary_checkpoints,
            "model_checkpoints": model_checkpoints,
        }
        atomic_json(phase_result, completion_path)
        phase_results.append(phase_result)
        previous_selected = selected

    if previous_selected is None:
        raise AssertionError("No final model checkpoint was produced")
    final_payload = _torch_load_cpu(previous_selected)
    final_path = output / "FINAL_MODEL.pt"
    _atomic_torch_save(
        {
            **final_payload,
            "source_checkpoint": str(previous_selected.relative_to(output)),
            "source_checkpoint_sha256": sha256_file(previous_selected),
        },
        final_path,
    )
    result = {
        "status": "complete",
        "experiment_id": bundle.experiment["experiment_id"],
        "method_hash": bundle.method_hash,
        "phases": phase_results,
        "end_global_step": global_step,
        "final_model": final_path.name,
        "final_model_sha256": sha256_file(final_path),
        "verified_points_used": False,
    }
    atomic_json(result, root_completion)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result

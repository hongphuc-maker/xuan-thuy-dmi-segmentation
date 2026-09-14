from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
from .selection import CheckpointSelector


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _build_schedule(maximum_steps: int, n_batches: int, seed: int) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    orders: dict[int, np.ndarray] = {}
    for zero_based_step in range(int(maximum_steps)):
        pass_index = zero_based_step // n_batches
        position = zero_based_step % n_batches
        if pass_index not in orders:
            orders[pass_index] = (
                np.arange(n_batches, dtype=np.int64)
                if pass_index == 0
                else np.random.default_rng(seed + pass_index).permutation(n_batches)
            )
        rows.append(
            {
                "optimizer_step": zero_based_step + 1,
                "pass_index": pass_index,
                "position_in_pass": position,
                "batch_index": int(orders[pass_index][position]),
            }
        )
    return rows


def _initialization_spec(phase: dict[str, Any]) -> dict[str, Any]:
    spec = phase.get("initialization")
    if not isinstance(spec, dict) or spec.get("type") != "checkpoint":
        raise ValueError("Training runs require phases[].initialization.type=checkpoint")
    expected = str(spec.get("sha256", ""))
    if len(expected) != 64:
        raise ValueError("Initialization checkpoint must have a pinned SHA-256")
    return spec


def _load_initialization(model: torch.nn.Module, path: Path, expected_sha256: str) -> dict[str, Any]:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"Initialization SHA-256 mismatch: expected {expected_sha256}, received {actual}"
        )
    payload = _torch_load_cpu(path)
    if not isinstance(payload, dict):
        raise ValueError("Initialization checkpoint must contain a mapping")
    state = payload.get("model_state", payload)
    model.load_state_dict(state, strict=True)
    return {"path": str(path), "sha256": actual, "payload_keys": sorted(payload)}


def _freeze_or_validate_run(
    bundle: ConfigBundle,
    data_root: Path,
    artifact_root: Path,
    initialization_path: Path,
    output: Path,
    runtime: dict[str, Any],
    data_verification: list[dict[str, Any]],
    cp2_verification: dict[str, Any],
) -> None:
    lock_path = output / "RUN_LOCK.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("method_hash") != bundle.method_hash:
            raise ValueError("Existing run directory was frozen with a different method hash")
        metadata_path = output / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if stable_hash(metadata) != lock.get("run_metadata_sha256"):
            raise ValueError("Existing run_metadata.json no longer matches RUN_LOCK.json")
        repository_root = Path(__file__).resolve().parents[3]
        if metadata.get("git_commit") != git_commit(repository_root):
            raise ValueError("Refusing to resume this run with a different source commit")
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Non-empty output directory has no RUN_LOCK.json: {output}")
    output.mkdir(parents=True, exist_ok=True)
    repository_root = Path(__file__).resolve().parents[3]
    metadata = {
        "experiment_id": bundle.experiment["experiment_id"],
        "method_hash": bundle.method_hash,
        "git_commit": git_commit(repository_root),
        "data_root": str(data_root),
        "artifact_root": str(artifact_root),
        "initialization_checkpoint": str(initialization_path),
        "initialization_sha256": sha256_file(initialization_path),
        "verified_files": data_verification,
        "cp2_artifact_files": cp2_verification["files"],
        "runtime": runtime,
    }
    atomic_json(bundle.resolved, output / "resolved_config.json")
    atomic_json(metadata, output / "run_metadata.json")
    atomic_json(
        {
            "status": "frozen",
            "experiment_id": bundle.experiment["experiment_id"],
            "method_hash": bundle.method_hash,
            "run_metadata_sha256": stable_hash(metadata),
        },
        lock_path,
    )


def _metrics_row(step: int, summary: dict[str, Any], **extra: Any) -> dict[str, Any]:
    row = {"optimizer_step": int(step), **extra}
    for key, value in summary.items():
        row[key] = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, dict) else value
    return row


def _save_tables(
    output: Path,
    history_rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
) -> None:
    _atomic_csv(pd.DataFrame(history_rows), output / "validation_history.csv")
    _atomic_csv(pd.DataFrame(per_class_rows), output / "validation_per_class.csv")
    _atomic_csv(pd.DataFrame(step_rows), output / "training_step_log.csv")


def _save_resume(
    output: Path,
    method_hash: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    accepted_selector: CheckpointSelector,
    unrestricted_selector: CheckpointSelector,
    history_rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
) -> None:
    _atomic_torch_save(
        {
            "method_hash": method_hash,
            "global_step": int(global_step),
            "model_state": _cpu_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "accepted_selector": accepted_selector.state_dict(),
            "unrestricted_selector": unrestricted_selector.state_dict(),
            "history_rows": history_rows,
            "per_class_rows": per_class_rows,
            "step_rows": step_rows,
        },
        output / "latest_resume_checkpoint.pt",
    )


def _save_best(
    path: Path,
    bundle: ConfigBundle,
    model: torch.nn.Module,
    step: int,
    validation: dict[str, Any],
    accepted: bool,
) -> None:
    _atomic_torch_save(
        {
            "method_hash": bundle.method_hash,
            "experiment_id": bundle.experiment["experiment_id"],
            "optimizer_step": int(step),
            "validation": validation,
            "acceptance_pass": bool(accepted),
            "model_config": bundle.model,
            "model_state": _cpu_state_dict(model),
        },
        path,
    )


def _prepare_loss(phase: dict[str, Any], data: CP2Data, device: torch.device):
    loss_config = phase["loss"]
    if loss_config["type"] == "ce_weighted":
        configured = np.asarray(
            loss_config["parameters"]["weighting"]["class_counts"],
            dtype=np.int64,
        )
        train_valid = data.common_valid & (data.split_mask == 1)
        observed = np.bincount(
            data.label[train_valid].astype(np.int64),
            minlength=len(data.class_map) + 1,
        )[1:]
        if not np.array_equal(configured, observed):
            raise ValueError(
                "Weighted-CE class_counts are not the unique train-core pixel counts: "
                f"configured={configured.tolist()}, observed={observed.tolist()}"
            )
    return build_loss(loss_config).to(device)


def train_experiment(
    bundle: ConfigBundle,
    data_root: str | Path,
    artifact_root: str | Path,
    initialization_checkpoint: str | Path,
    output: str | Path,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Run the single-phase CP2-backed training protocol with automatic resume."""
    if len(bundle.resolved["phases"]) != 1:
        raise ValueError("The current runner supports exactly one training phase")
    phase = bundle.resolved["phases"][0]
    initialization = _initialization_spec(phase)
    data_root = Path(data_root).resolve()
    artifact_root = Path(artifact_root).resolve()
    initialization_path = Path(initialization_checkpoint).resolve()
    output = Path(output).resolve()
    if not initialization_path.is_file():
        raise FileNotFoundError(initialization_path)

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is available")
    runtime = configure_reproducibility(int(bundle.experiment["seed"]), warn_only=False)
    data_verification = verify_files(data_root, bundle.dataset["files"])
    data = load_cp2_data(bundle.dataset, bundle.split, data_root, artifact_root)
    _freeze_or_validate_run(
        bundle,
        data_root,
        artifact_root,
        initialization_path,
        output,
        runtime,
        data_verification,
        data.verification,
    )

    completion_path = output / "TRAINING_COMPLETE.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("method_hash") != bundle.method_hash:
            raise ValueError("Completion marker has a different method hash")
        print(f"Run already complete: {completion_path}")
        return completion

    model = build_model(bundle.model).to(device)
    initialization_metadata = _load_initialization(
        model,
        initialization_path,
        str(initialization["sha256"]),
    )
    optimizer = build_optimizer(model.parameters(), phase["optimizer"])
    loss_function = _prepare_loss(phase, data, device)
    maximum_steps = int(phase["maximum_steps"])
    validation_interval = int(phase.get("validation_interval_steps", 156))
    monitor_interval = int(phase.get("monitor_interval_steps", 10))
    schedule = _build_schedule(maximum_steps, len(data.frozen_batches), int(bundle.experiment["seed"]))
    schedule_frame = pd.DataFrame(schedule)
    schedule_path = output / "global_step_batch_schedule.csv"
    if schedule_path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(schedule_path), schedule_frame)
    else:
        _atomic_csv(schedule_frame, schedule_path)

    accepted_selector = CheckpointSelector(bundle.experiment["selection"], bundle.experiment["acceptance"])
    unrestricted_selector = CheckpointSelector(bundle.experiment["selection"], {})
    latest_path = output / "latest_resume_checkpoint.pt"
    history_rows: list[dict[str, Any]]
    per_class_rows: list[dict[str, Any]]
    step_rows: list[dict[str, Any]]
    if latest_path.exists():
        resume = _torch_load_cpu(latest_path)
        if resume["method_hash"] != bundle.method_hash:
            raise ValueError("Resume checkpoint has a different method hash")
        model.load_state_dict(resume["model_state"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state"])
        global_step = int(resume["global_step"])
        accepted_selector.load_state_dict(resume["accepted_selector"])
        unrestricted_selector.load_state_dict(resume["unrestricted_selector"])
        history_rows = list(resume["history_rows"])
        per_class_rows = list(resume["per_class_rows"])
        step_rows = list(resume["step_rows"])
        print(f"AUTO-RESUME: optimizer step {global_step}/{maximum_steps}")
    else:
        global_step = 0
        history_rows, per_class_rows, step_rows = [], [], []
        summary, per_class, _ = evaluate_spatial_validation(model, data, device)
        unrestricted = unrestricted_selector.update(global_step, summary)
        accepted = accepted_selector.update(global_step, summary)
        history_rows.append(
            _metrics_row(
                global_step,
                summary,
                accepted=accepted.eligible,
                improved_accepted=accepted.improved,
                improved_unrestricted=unrestricted.improved,
                no_improve_accepted=accepted.no_improve_evaluations,
            )
        )
        per_class.insert(0, "optimizer_step", global_step)
        per_class_rows.extend(json.loads(per_class.to_json(orient="records")))
        _save_best(
            output / "best_unrestricted_checkpoint.pt",
            bundle,
            model,
            global_step,
            summary,
            accepted=False,
        )
        if accepted.improved:
            _save_best(
                output / "best_accepted_checkpoint.pt",
                bundle,
                model,
                global_step,
                summary,
                accepted=True,
            )
        _save_tables(output, history_rows, per_class_rows, step_rows)
        _save_resume(
            output,
            bundle.method_hash,
            model,
            optimizer,
            global_step,
            accepted_selector,
            unrestricted_selector,
            history_rows,
            per_class_rows,
            step_rows,
        )

    class_weights = getattr(loss_function, "class_weights", None)
    if class_weights is not None:
        weights = [float(value) for value in class_weights.detach().cpu()]
        print("Weighted CE class weights:", [round(value, 6) for value in weights])
        atomic_json(
            {
                "loss_id": phase["loss"]["loss_id"],
                "class_codes": list(sorted(data.class_map)),
                "class_weights": weights,
            },
            output / "loss_runtime.json",
        )
    print(
        f"TRAIN {bundle.experiment['experiment_id']} | {global_step}->{maximum_steps} steps | "
        f"validation every {validation_interval} steps | device={device}"
    )

    recent_losses: deque[float] = deque(maxlen=monitor_interval)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    progress = tqdm(
        total=maximum_steps,
        initial=global_step,
        desc=str(bundle.experiment["experiment_id"]),
        unit="step",
        dynamic_ncols=True,
        mininterval=1.0,
    )
    early_stopped = False
    interrupted = False
    started = time.perf_counter()
    session_start_step = global_step
    try:
        while global_step < maximum_steps:
            schedule_row = schedule[global_step]
            batch = data.materialize_batch(schedule_row["batch_index"])
            inputs = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_function(logits, targets)
            if not bool(torch.isfinite(loss).detach().cpu()):
                raise FloatingPointError(f"Non-finite loss before step {global_step + 1}")
            loss.backward()
            nonfinite_gradients = sum(
                int((~torch.isfinite(parameter.grad)).sum().detach().cpu())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            if nonfinite_gradients:
                raise FloatingPointError(f"Non-finite gradients before step {global_step + 1}")
            optimizer.step()
            global_step += 1
            loss_value = float(loss.detach().cpu())
            recent_losses.append(loss_value)
            step_rows.append(
                {
                    **schedule_row,
                    "loss": loss_value,
                    "nonfinite_gradient_values": nonfinite_gradients,
                }
            )
            progress.update(1)
            if global_step % monitor_interval == 0 or global_step == maximum_steps:
                postfix = {
                    "loss10": f"{np.mean(recent_losses):.4f}",
                    "accepted": "yes" if accepted_selector.best_step is not None else "no",
                }
                if unrestricted_selector.best_value is not None:
                    postfix["bestF1"] = f"{unrestricted_selector.best_value:.4f}"
                if device.type == "cuda":
                    postfix.update(
                        {
                            "GPU": f"{torch.cuda.memory_allocated(device) / 2**30:.2f}G",
                            "peak": f"{torch.cuda.max_memory_allocated(device) / 2**30:.2f}G",
                        }
                    )
                progress.set_postfix(postfix, refresh=True)
            del batch, inputs, targets, logits, loss

            if global_step % validation_interval != 0 and global_step != maximum_steps:
                continue
            progress.write(f"Validation started at step {global_step}...")
            evaluation_started = time.perf_counter()
            summary, per_class, _ = evaluate_spatial_validation(model, data, device)
            evaluation_seconds = time.perf_counter() - evaluation_started
            unrestricted = unrestricted_selector.update(global_step, summary)
            accepted = accepted_selector.update(global_step, summary)
            history_rows.append(
                _metrics_row(
                    global_step,
                    summary,
                    accepted=accepted.eligible,
                    acceptance_reason=accepted.reason,
                    improved_accepted=accepted.improved,
                    improved_unrestricted=unrestricted.improved,
                    no_improve_accepted=accepted.no_improve_evaluations,
                )
            )
            per_class.insert(0, "optimizer_step", global_step)
            per_class_rows.extend(json.loads(per_class.to_json(orient="records")))
            if unrestricted.improved:
                _save_best(
                    output / "best_unrestricted_checkpoint.pt",
                    bundle,
                    model,
                    global_step,
                    summary,
                    accepted=False,
                )
            if accepted.improved:
                _save_best(
                    output / "best_accepted_checkpoint.pt",
                    bundle,
                    model,
                    global_step,
                    summary,
                    accepted=True,
                )
            _save_tables(output, history_rows, per_class_rows, step_rows)
            _save_resume(
                output,
                bundle.method_hash,
                model,
                optimizer,
                global_step,
                accepted_selector,
                unrestricted_selector,
                history_rows,
                per_class_rows,
                step_rows,
            )
            progress.write(
                f"step={global_step}/{maximum_steps} "
                f"OA={summary['validation_oa']:.4f} "
                f"macro-F1={summary['validation_macro_f1_11']:.4f} "
                f"macro-IoU={summary['validation_macro_iou_11']:.4f} "
                f"predicted={summary['predicted_classes']}/11 "
                f"accepted={accepted.eligible} "
                f"bestAccepted={accepted.best_value if accepted.best_value is not None else '—'} "
                f"patience={accepted.no_improve_evaluations}/{accepted_selector.patience} "
                f"validation={evaluation_seconds:.1f}s"
            )
            if accepted.should_stop:
                early_stopped = True
                break
    except KeyboardInterrupt:
        interrupted = True
        progress.write("Training interrupted; saving latest_resume_checkpoint.pt...")
    finally:
        progress.close()
        _save_tables(output, history_rows, per_class_rows, step_rows)
        _save_resume(
            output,
            bundle.method_hash,
            model,
            optimizer,
            global_step,
            accepted_selector,
            unrestricted_selector,
            history_rows,
            per_class_rows,
            step_rows,
        )

    elapsed = time.perf_counter() - started
    session_steps = global_step - session_start_step
    if session_steps:
        print(
            f"Session completed {session_steps} steps in {elapsed / 60:.1f} minutes "
            f"({elapsed / session_steps:.2f} seconds/step)."
        )
    if interrupted:
        return {
            "status": "interrupted_resume_available",
            "method_hash": bundle.method_hash,
            "optimizer_step": global_step,
            "resume_checkpoint": str(latest_path),
        }

    accepted_path = output / "best_accepted_checkpoint.pt"
    unrestricted_path = output / "best_unrestricted_checkpoint.pt"
    selected_path = accepted_path if accepted_path.exists() else unrestricted_path
    result = {
        "status": "complete",
        "experiment_id": bundle.experiment["experiment_id"],
        "method_hash": bundle.method_hash,
        "execution_status": "early_stopped" if early_stopped else "max_steps_reached",
        "end_optimizer_step": global_step,
        "acceptance_pass": accepted_path.exists(),
        "best_accepted_step": accepted_selector.best_step,
        "best_accepted_value": accepted_selector.best_value,
        "best_unrestricted_step": unrestricted_selector.best_step,
        "best_unrestricted_value": unrestricted_selector.best_value,
        "selected_model": selected_path.name,
        "selected_model_sha256": sha256_file(selected_path),
        "latest_resume_checkpoint_sha256": sha256_file(latest_path),
        "schedule_sha256": sha256_file(schedule_path),
        "initialization": initialization_metadata,
        "verified_points_used": False,
    }
    atomic_json(result, completion_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result

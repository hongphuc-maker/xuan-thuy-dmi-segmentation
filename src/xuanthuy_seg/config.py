from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from .contracts import stable_hash


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_SPLITS = {"vertical_band", "horizontal_band", "spatial_block_kfold", "external_mask"}
ALLOWED_LOSSES = {"ce_unweighted", "ce_weighted", "dmi_exact", "dmi_regularized"}
ALLOWED_OPTIMIZERS = {"sgd", "adam", "adamw"}
ALLOWED_DIRECTIONS = {"min", "max"}
ALLOWED_STOP_RULES = {"patience", "maximum_steps"}


@dataclass(frozen=True)
class ConfigBundle:
    experiment_path: Path
    experiment: dict[str, Any]
    dataset: dict[str, Any]
    split: dict[str, Any]
    model: dict[str, Any]
    losses: dict[str, dict[str, Any]]
    resolved: dict[str, Any]
    method_hash: str


@dataclass(frozen=True)
class TemporalTransferBundle:
    transfer_path: Path
    transfer: dict[str, Any]
    source: ConfigBundle
    target: ConfigBundle
    resolved: dict[str, Any]
    transfer_hash: str


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return _normalize_yaml(payload)


def _normalize_yaml(value: Any) -> Any:
    """Normalize YAML-specific scalar types before hashing/provenance export."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _normalize_yaml(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_yaml(item) for item in value]
    return value


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _require(mapping: dict[str, Any], keys: set[str], context: str) -> None:
    missing = sorted(keys - mapping.keys())
    if missing:
        raise ValueError(f"{context} is missing keys: {missing}")


def _validate_date(value: Any, context: str) -> None:
    try:
        date.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError(f"{context} must use YYYY-MM-DD: {value}") from error


def validate_dataset(config: dict[str, Any]) -> None:
    _require(config, {"schema_version", "dataset_id", "files", "bands", "class_map"}, "dataset")
    if len(config["bands"]) != 10:
        raise ValueError("The current model contract requires exactly 10 Sentinel-2 bands")
    class_map = {int(key): str(value) for key, value in config["class_map"].items()}
    if sorted(class_map) != list(range(1, 12)):
        raise ValueError("class_map must contain source codes 1..11 exactly")

    roles: set[str] = set()
    for spec in config["files"]:
        _require(spec, {"role", "path", "sha256"}, "dataset.files item")
        role = str(spec["role"])
        if role in roles:
            raise ValueError(f"Duplicate dataset file role: {role}")
        roles.add(role)
        sha = str(spec["sha256"]).lower()
        if sha and not SHA256_RE.fullmatch(sha):
            raise ValueError(f"Invalid SHA-256 for role {role}")
        if "acquired_at" in spec:
            _validate_date(spec["acquired_at"], f"{role}.acquired_at")
        if "reference_date" in spec:
            _validate_date(spec["reference_date"], f"{role}.reference_date")
    if not {"image", "label"}.issubset(roles):
        raise ValueError("dataset files must define at least image and label roles")


def validate_split(config: dict[str, Any]) -> None:
    _require(config, {"schema_version", "split_id", "strategy", "parameters"}, "split")
    strategy = str(config["strategy"])
    if strategy not in ALLOWED_SPLITS:
        raise ValueError(f"Unsupported spatial split strategy: {strategy}")
    if strategy in {"vertical_band", "horizontal_band"}:
        _require(
            config["parameters"],
            {"validation_start", "validation_stop", "guard_px_each_side"},
            "split.parameters",
        )
    if strategy == "spatial_block_kfold":
        _require(
            config["parameters"],
            {"block_height", "block_width", "n_folds", "validation_fold", "seed", "guard_px"},
            "split.parameters",
        )
        parameters = config["parameters"]
        if min(int(parameters["block_height"]), int(parameters["block_width"])) <= 0:
            raise ValueError("spatial block dimensions must be positive")
        if int(parameters["n_folds"]) < 2:
            raise ValueError("spatial_block_kfold requires n_folds >= 2")
        if not 0 <= int(parameters["validation_fold"]) < int(parameters["n_folds"]):
            raise ValueError("validation_fold must be in [0, n_folds)")
    if "patch_policy" in config or "sampling" in config:
        _require(config, {"patch_policy", "sampling"}, "generated split")
        _require(config["patch_policy"], {"size"}, "split.patch_policy")
        _require(
            config["sampling"],
            {"manifest_seed", "schedule_seed", "n_patches", "batch_size"},
            "split.sampling",
        )
        patch_size = int(config["patch_policy"]["size"])
        n_patches = int(config["sampling"]["n_patches"])
        batch_size = int(config["sampling"]["batch_size"])
        if patch_size <= 0 or patch_size % 2:
            raise ValueError("patch_policy.size must be a positive even number")
        if n_patches <= 0 or batch_size <= 0 or n_patches % batch_size:
            raise ValueError("n_patches must be positive and divisible by batch_size")
    artifact_contract = config.get("artifact_contract")
    if artifact_contract is not None:
        _require(artifact_contract, {"type", "data_method_hash", "files_sha256"}, "split.artifact_contract")
        if not SHA256_RE.fullmatch(str(artifact_contract["data_method_hash"])):
            raise ValueError("split.artifact_contract.data_method_hash must be SHA-256")
        for name, value in artifact_contract["files_sha256"].items():
            if not SHA256_RE.fullmatch(str(value)):
                raise ValueError(f"Invalid CP2 artifact SHA-256 for {name}")


def validate_model(config: dict[str, Any]) -> None:
    _require(config, {"schema_version", "model_id", "architecture", "parameters", "output"}, "model")
    if config["output"] not in {"logits", "probabilities"}:
        raise ValueError("model.output must be logits or probabilities")
    if config["architecture"] != "unet_bn" and not config.get("factory"):
        raise ValueError(f"Unsupported model architecture: {config['architecture']}")
    parameters = config["parameters"]
    if config.get("factory"):
        if ":" not in str(config["factory"]):
            raise ValueError("Custom model factory must use 'module:callable'")
        return
    allowed = {"in_channels", "num_classes", "base_channels"}
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise ValueError(f"Unknown UNetBN constructor parameters: {unknown}")
    _require(parameters, allowed, "model.parameters")
    if int(parameters["in_channels"]) != 10 or int(parameters["num_classes"]) != 11:
        raise ValueError("Current data/model contract requires 10 input bands and 11 classes")


def validate_loss(config: dict[str, Any]) -> None:
    _require(config, {"schema_version", "loss_id", "type", "parameters"}, "loss")
    if config["type"] not in ALLOWED_LOSSES and not config.get("factory"):
        raise ValueError(f"Unsupported loss type: {config['type']}")
    if config.get("factory") and ":" not in str(config["factory"]):
        raise ValueError("Custom loss factory must use 'module:callable'")
    if config.get("factory") and config.get("input") not in {"logits", "probabilities"}:
        raise ValueError("Custom loss must declare input as logits or probabilities")
    if config["type"] == "ce_weighted":
        weighting = config["parameters"].get("weighting", {})
        if weighting.get("method") not in {"sqrt_inverse_frequency", "effective_number", "explicit"}:
            raise ValueError("Weighted CE requires a supported weighting.method")


def _validate_selection(selection: dict[str, Any], context: str) -> None:
    _require(selection, {"checkpoint", "stopping"}, context)
    checkpoint = selection["checkpoint"]
    _require(checkpoint, {"metric", "direction", "min_delta"}, f"{context}.checkpoint")
    if checkpoint["direction"] not in ALLOWED_DIRECTIONS:
        raise ValueError(f"{context}.checkpoint.direction must be min or max")
    stopping = selection["stopping"]
    _require(stopping, {"rule"}, f"{context}.stopping")
    if stopping["rule"] not in ALLOWED_STOP_RULES:
        raise ValueError(f"{context}.stopping.rule must be patience or maximum_steps")
    if stopping["rule"] == "patience" and int(stopping.get("patience_evaluations", 0)) <= 0:
        raise ValueError(f"{context} patience stopping requires patience_evaluations > 0")


def validate_experiment(config: dict[str, Any]) -> None:
    _require(
        config,
        {"schema_version", "experiment_id", "seed", "dataset", "split", "model", "phases", "selection", "acceptance"},
        "experiment",
    )
    if not config["phases"]:
        raise ValueError("experiment.phases cannot be empty")
    phase_ids: set[str] = set()
    for phase in config["phases"]:
        _require(phase, {"phase_id", "loss", "optimizer", "maximum_steps"}, "training phase")
        phase_id = str(phase["phase_id"])
        if phase_id in phase_ids:
            raise ValueError(f"Duplicate phase_id: {phase_id}")
        phase_ids.add(phase_id)
        optimizer = phase["optimizer"]
        _require(optimizer, {"type", "learning_rate"}, f"optimizer for {phase_id}")
        if optimizer["type"] not in ALLOWED_OPTIMIZERS:
            raise ValueError(f"Unsupported optimizer: {optimizer['type']}")
        if float(optimizer["learning_rate"]) <= 0:
            raise ValueError(f"learning_rate must be positive in phase {phase_id}")
        if int(phase["maximum_steps"]) <= 0:
            raise ValueError(f"maximum_steps must be positive in phase {phase_id}")
        if int(phase.get("validation_interval_steps", 1)) <= 0:
            raise ValueError(f"validation_interval_steps must be positive in phase {phase_id}")
        if int(phase.get("monitor_interval_steps", 1)) <= 0:
            raise ValueError(f"monitor_interval_steps must be positive in phase {phase_id}")
        if phase.get("handoff_checkpoint", "selected") not in {"selected", "latest"}:
            raise ValueError(f"handoff_checkpoint must be selected or latest in phase {phase_id}")
        if "selection" in phase:
            _validate_selection(phase["selection"], f"selection for phase {phase_id}")
        initialization = phase.get("initialization")
        if isinstance(initialization, dict):
            _require(initialization, {"type", "sha256"}, f"initialization for {phase_id}")
            if initialization["type"] != "checkpoint":
                raise ValueError(f"Unsupported initialization type in phase {phase_id}")
            if not SHA256_RE.fullmatch(str(initialization["sha256"])):
                raise ValueError(f"Invalid initialization SHA-256 in phase {phase_id}")
            target_module = str(initialization.get("target_module", "model"))
            if not target_module or any(
                not part.isidentifier() for part in target_module.split(".")
            ):
                raise ValueError(
                    f"Invalid initialization target_module in phase {phase_id}: "
                    f"{target_module}"
                )

    _validate_selection(config["selection"], "selection")


def load_experiment_bundle(path: str | Path) -> ConfigBundle:
    experiment_path = Path(path).resolve()
    experiment = _read_yaml(experiment_path)
    validate_experiment(experiment)
    base = experiment_path.parent

    dataset_path = _resolve(base, str(experiment["dataset"]))
    split_path = _resolve(base, str(experiment["split"]))
    model_path = _resolve(base, str(experiment["model"]))
    dataset = _read_yaml(dataset_path)
    split = _read_yaml(split_path)
    model = _read_yaml(model_path)
    validate_dataset(dataset)
    validate_split(split)
    validate_model(model)

    losses: dict[str, dict[str, Any]] = {}
    resolved_phases = copy.deepcopy(experiment["phases"])
    for original, resolved_phase in zip(experiment["phases"], resolved_phases, strict=True):
        loss_path = _resolve(base, str(original["loss"]))
        loss_config = _read_yaml(loss_path)
        validate_loss(loss_config)
        expected_input = str(loss_config.get("input", "logits"))
        if expected_input != str(model["output"]):
            raise ValueError(
                f"Phase {original['phase_id']} loss expects {expected_input}, "
                f"but model outputs {model['output']}"
            )
        loss_id = str(loss_config["loss_id"])
        losses[loss_id] = loss_config
        resolved_phase["loss"] = loss_config

    resolved = copy.deepcopy(experiment)
    resolved["dataset"] = dataset
    resolved["split"] = split
    resolved["model"] = model
    resolved["phases"] = resolved_phases
    method_hash = stable_hash(resolved)
    resolved["method_hash"] = method_hash

    return ConfigBundle(
        experiment_path=experiment_path,
        experiment=experiment,
        dataset=dataset,
        split=split,
        model=model,
        losses=losses,
        resolved=resolved,
        method_hash=method_hash,
    )


def load_temporal_transfer_bundle(path: str | Path) -> TemporalTransferBundle:
    transfer_path = Path(path).resolve()
    transfer = _read_yaml(transfer_path)
    _require(
        transfer,
        {
            "schema_version",
            "transfer_id",
            "source_experiment",
            "target_experiment",
            "normalization_policy",
            "evaluation_interpretation",
        },
        "temporal transfer",
    )
    if transfer["schema_version"] != "xtseg-temporal-transfer-v1":
        raise ValueError("Unsupported temporal-transfer schema_version")
    if transfer["normalization_policy"] != "source_training_artifact":
        raise ValueError(
            "Temporal transfer must use source_training_artifact normalization"
        )

    base = transfer_path.parent
    source = load_experiment_bundle(_resolve(base, str(transfer["source_experiment"])))
    target = load_experiment_bundle(_resolve(base, str(transfer["target_experiment"])))
    if source.model != target.model:
        raise ValueError("Source and target experiments must use the same model contract")
    if source.dataset["bands"] != target.dataset["bands"]:
        raise ValueError("Source and target datasets must use the same ordered bands")
    if source.dataset["class_map"] != target.dataset["class_map"]:
        raise ValueError("Source and target datasets must use the same class map")

    resolved = copy.deepcopy(transfer)
    resolved["source"] = {
        "experiment_id": source.experiment["experiment_id"],
        "method_hash": source.method_hash,
        "dataset": source.dataset,
        "split": source.split,
        "model": source.model,
    }
    resolved["target"] = {
        "dataset": target.dataset,
        "split": target.split,
        "model": target.model,
    }
    resolved.pop("source_experiment", None)
    resolved.pop("target_experiment", None)
    transfer_hash = stable_hash(resolved)
    resolved["transfer_hash"] = transfer_hash
    return TemporalTransferBundle(
        transfer_path=transfer_path,
        transfer=transfer,
        source=source,
        target=target,
        resolved=resolved,
        transfer_hash=transfer_hash,
    )

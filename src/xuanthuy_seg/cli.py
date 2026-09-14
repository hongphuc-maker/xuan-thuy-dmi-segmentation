from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import ConfigBundle, load_experiment_bundle, load_temporal_transfer_bundle
from .contracts import atomic_json, stable_hash, verify_files
from .reproducibility import git_commit


def _summary(bundle: ConfigBundle) -> dict[str, Any]:
    return {
        "experiment_id": bundle.experiment["experiment_id"],
        "dataset_id": bundle.dataset["dataset_id"],
        "split_id": bundle.split["split_id"],
        "model_id": bundle.model["model_id"],
        "phases": [
            {
                "phase_id": phase["phase_id"],
                "loss_id": phase["loss"]["loss_id"],
                "optimizer": phase["optimizer"],
                "maximum_steps": phase["maximum_steps"],
                "initialization": phase.get("initialization"),
            }
            for phase in bundle.resolved["phases"]
        ],
        "checkpoint_selection": bundle.experiment["selection"]["checkpoint"],
        "stopping": bundle.experiment["selection"]["stopping"],
        "acceptance": bundle.experiment["acceptance"],
        "method_hash": bundle.method_hash,
    }


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def command_validate_config(args: argparse.Namespace) -> int:
    bundle = load_experiment_bundle(args.experiment)
    _print_json(_summary(bundle))
    return 0


def command_verify_data(args: argparse.Namespace) -> int:
    bundle = load_experiment_bundle(args.experiment)
    requested_roles = set(args.roles or [])
    file_specs = [
        spec for spec in bundle.dataset["files"]
        if not requested_roles or str(spec["role"]) in requested_roles
    ]
    available_roles = {str(spec["role"]) for spec in bundle.dataset["files"]}
    missing_roles = sorted(requested_roles - available_roles)
    if missing_roles:
        raise ValueError(f"Dataset config does not define requested roles: {missing_roles}")
    results = verify_files(args.data_root, file_specs)
    _print_json(
        {
            "status": "pass",
            "dataset_id": bundle.dataset["dataset_id"],
            "verified_roles": [str(spec["role"]) for spec in file_specs],
            "files": results,
        }
    )
    return 0


def command_inspect_inputs(args: argparse.Namespace) -> int:
    from .data.inspection import inspect_inputs

    _print_json(
        inspect_inputs(
            args.data_root,
            args.image,
            args.label,
            args.verified_points,
            args.expected_bands,
        )
    )
    return 0


def command_freeze_run(args: argparse.Namespace) -> int:
    bundle = load_experiment_bundle(args.experiment)
    output = Path(args.output).resolve()
    lock_path = output / "RUN_LOCK.json"
    if lock_path.exists():
        raise FileExistsError(
            f"Run is already frozen at {output}; choose a new output directory"
        )
    verification = verify_files(args.data_root, bundle.dataset["files"])
    repository_root = Path(__file__).resolve().parents[2]
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(bundle.resolved, output / "resolved_config.json")
    metadata = {
        **_summary(bundle),
        "git_commit": git_commit(repository_root),
        "data_root": str(Path(args.data_root).resolve()),
        "verified_files": verification,
    }
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
    _print_json({"status": "frozen", "output": str(output), **_summary(bundle)})
    return 0


def command_train(args: argparse.Namespace) -> int:
    # Import lazily so validate-config/verify-data do not require the training extras.
    from .training.experiment import train_pipeline_experiment

    bundle = load_experiment_bundle(args.experiment)
    result = train_pipeline_experiment(
        bundle=bundle,
        data_root=args.data_root,
        artifact_root=args.artifact_root,
        initialization_checkpoint=args.initialization_checkpoint,
        output=args.output,
        device_name=args.device,
    )
    _print_json(result)
    return 0


def command_prepare_data(args: argparse.Namespace) -> int:
    from .data.preparation import prepare_data

    bundle = load_experiment_bundle(args.experiment)
    _print_json(prepare_data(bundle, args.data_root, args.output))
    return 0


def command_infer_map(args: argparse.Namespace) -> int:
    from .inference import infer_full_map

    bundle = load_experiment_bundle(args.experiment)
    _print_json(
        infer_full_map(
            bundle,
            args.data_root,
            args.artifact_root,
            args.checkpoint,
            args.output,
            args.device,
        )
    )
    return 0


def command_evaluate_points(args: argparse.Namespace) -> int:
    if not args.confirm_independent_evaluation:
        raise ValueError(
            "Independent points are sealed from model selection. Re-run with "
            "--confirm-independent-evaluation only after the model/protocol is final."
        )
    from .independent_evaluation import evaluate_verified_points

    bundle = load_experiment_bundle(args.experiment)
    _print_json(evaluate_verified_points(bundle, args.data_root, args.map, args.output))
    return 0


def command_validate_temporal_transfer(args: argparse.Namespace) -> int:
    bundle = load_temporal_transfer_bundle(args.transfer)
    _print_json(
        {
            "transfer_id": bundle.transfer["transfer_id"],
            "transfer_hash": bundle.transfer_hash,
            "source_experiment_id": bundle.source.experiment["experiment_id"],
            "source_method_hash": bundle.source.method_hash,
            "source_dataset_id": bundle.source.dataset["dataset_id"],
            "target_dataset_id": bundle.target.dataset["dataset_id"],
            "normalization_policy": bundle.transfer["normalization_policy"],
            "evaluation_interpretation": bundle.transfer["evaluation_interpretation"],
        }
    )
    return 0


def command_infer_temporal_transfer(args: argparse.Namespace) -> int:
    from .temporal_transfer import infer_temporal_transfer_map

    bundle = load_temporal_transfer_bundle(args.transfer)
    _print_json(
        infer_temporal_transfer_map(
            bundle,
            args.data_root,
            args.source_artifact_root,
            args.checkpoint,
            args.output,
            args.device,
        )
    )
    return 0


def command_evaluate_temporal_transfer(args: argparse.Namespace) -> int:
    from .temporal_transfer import evaluate_temporal_transfer_points

    if not args.confirm_independent_evaluation:
        raise ValueError(
            "Temporal verified points are sealed from model selection. Re-run with "
            "--confirm-independent-evaluation only after the source model is frozen."
        )
    bundle = load_temporal_transfer_bundle(args.transfer)
    _print_json(
        evaluate_temporal_transfer_points(
            bundle,
            args.data_root,
            args.map,
            args.output,
        )
    )
    return 0


def command_run_pipeline(args: argparse.Namespace) -> int:
    from .data.preparation import prepare_data
    from .independent_evaluation import evaluate_verified_points
    from .inference import infer_full_map
    from .training.experiment import train_pipeline_experiment

    if args.through == "evaluate" and not args.confirm_independent_evaluation:
        raise ValueError(
            "--through evaluate requires --confirm-independent-evaluation after model selection is frozen"
        )
    bundle = load_experiment_bundle(args.experiment)
    root = Path(args.run_root).resolve()
    prepared = root / "data_artifacts"
    training = root / "training"
    maps = root / "map"
    evaluation = root / "independent_evaluation"
    result: dict[str, Any] = {
        "data": prepare_data(bundle, args.data_root, prepared),
    }
    if args.through == "data":
        _print_json(result)
        return 0
    result["training"] = train_pipeline_experiment(
        bundle,
        args.data_root,
        prepared,
        training,
        args.device,
        args.initialization_checkpoint,
    )
    if result["training"].get("status") != "complete" or args.through == "train":
        _print_json(result)
        return 0
    result["map"] = infer_full_map(
        bundle,
        args.data_root,
        prepared,
        training / "FINAL_MODEL.pt",
        maps,
        args.device,
    )
    if args.through == "map":
        _print_json(result)
        return 0
    result["evaluation"] = evaluate_verified_points(
        bundle,
        args.data_root,
        maps / "prediction.tif",
        evaluation,
    )
    _print_json(result)
    return 0


def command_plot_training_history(args: argparse.Namespace) -> int:
    from .reporting import plot_ce_to_dmi_history

    _print_json(
        plot_ce_to_dmi_history(
            args.ce_run_root,
            args.dmi_run_root,
            args.output,
            args.transition_step,
            args.rolling_steps,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xtseg")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="Resolve and validate an experiment YAML")
    validate.add_argument("--experiment", required=True)
    validate.set_defaults(func=command_validate_config)

    verify = subparsers.add_parser("verify-data", help="Verify dataset files against SHA-256 contract")
    verify.add_argument("--experiment", required=True)
    verify.add_argument("--data-root", required=True)
    verify.add_argument(
        "--roles",
        nargs="+",
        help="Optional dataset roles to verify, for example: --roles image label",
    )
    verify.set_defaults(func=command_verify_data)

    inspect = subparsers.add_parser(
        "inspect-inputs",
        help="Inspect hashes and grid alignment before adding a dataset YAML",
    )
    inspect.add_argument("--data-root", required=True)
    inspect.add_argument("--image", required=True)
    inspect.add_argument("--label", required=True)
    inspect.add_argument("--verified-points")
    inspect.add_argument("--expected-bands", type=int, default=10)
    inspect.set_defaults(func=command_inspect_inputs)

    freeze = subparsers.add_parser("freeze-run", help="Seal config, data hashes and Git commit before training")
    freeze.add_argument("--experiment", required=True)
    freeze.add_argument("--data-root", required=True)
    freeze.add_argument("--output", required=True)
    freeze.set_defaults(func=command_freeze_run)

    prepare = subparsers.add_parser(
        "prepare-data",
        help="Generate checksummed split, patch, schedule, normalization and validation artifacts",
    )
    prepare.add_argument("--experiment", required=True)
    prepare.add_argument("--data-root", required=True)
    prepare.add_argument("--output", required=True)
    prepare.set_defaults(func=command_prepare_data)

    train = subparsers.add_parser(
        "train",
        help="Train a source-prepared experiment with monitoring and automatic resume",
    )
    train.add_argument("--experiment", required=True)
    train.add_argument("--data-root", required=True)
    train.add_argument("--artifact-root", required=True)
    train.add_argument("--initialization-checkpoint")
    train.add_argument("--output", required=True)
    train.add_argument("--device", default="cuda")
    train.set_defaults(func=command_train)

    infer = subparsers.add_parser("infer-map", help="Create and seal a full-scene GeoTIFF map")
    infer.add_argument("--experiment", required=True)
    infer.add_argument("--data-root", required=True)
    infer.add_argument("--artifact-root", required=True)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--device", default="cuda")
    infer.set_defaults(func=command_infer_map)

    evaluate = subparsers.add_parser(
        "evaluate-points",
        help="Evaluate a sealed map once against independent verified points",
    )
    evaluate.add_argument("--experiment", required=True)
    evaluate.add_argument("--data-root", required=True)
    evaluate.add_argument("--map", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--confirm-independent-evaluation", action="store_true")
    evaluate.set_defaults(func=command_evaluate_points)

    validate_transfer = subparsers.add_parser(
        "validate-temporal-transfer",
        help="Resolve a source-model to target-date transfer contract",
    )
    validate_transfer.add_argument("--transfer", required=True)
    validate_transfer.set_defaults(func=command_validate_temporal_transfer)

    infer_transfer = subparsers.add_parser(
        "infer-temporal-transfer",
        help="Apply a sealed source model to a target-date image using source normalization",
    )
    infer_transfer.add_argument("--transfer", required=True)
    infer_transfer.add_argument("--data-root", required=True)
    infer_transfer.add_argument("--source-artifact-root", required=True)
    infer_transfer.add_argument("--checkpoint", required=True)
    infer_transfer.add_argument("--output", required=True)
    infer_transfer.add_argument("--device", default="cuda")
    infer_transfer.set_defaults(func=command_infer_temporal_transfer)

    evaluate_transfer = subparsers.add_parser(
        "evaluate-temporal-transfer",
        help="Evaluate a sealed temporal-transfer map against verified points",
    )
    evaluate_transfer.add_argument("--transfer", required=True)
    evaluate_transfer.add_argument("--data-root", required=True)
    evaluate_transfer.add_argument("--map", required=True)
    evaluate_transfer.add_argument("--output", required=True)
    evaluate_transfer.add_argument("--confirm-independent-evaluation", action="store_true")
    evaluate_transfer.set_defaults(func=command_evaluate_temporal_transfer)

    pipeline = subparsers.add_parser(
        "run-pipeline",
        help="Prepare data, train/resume, and optionally create/evaluate the final map",
    )
    pipeline.add_argument("--experiment", required=True)
    pipeline.add_argument("--data-root", required=True)
    pipeline.add_argument("--run-root", required=True)
    pipeline.add_argument("--through", choices=["data", "train", "map", "evaluate"], default="train")
    pipeline.add_argument("--device", default="cuda")
    pipeline.add_argument("--initialization-checkpoint")
    pipeline.add_argument("--confirm-independent-evaluation", action="store_true")
    pipeline.set_defaults(func=command_run_pipeline)

    plot_history = subparsers.add_parser(
        "plot-training-history",
        help="Plot a CE-to-DMI trajectory from two completed/resumable run roots",
    )
    plot_history.add_argument("--ce-run-root", required=True)
    plot_history.add_argument("--dmi-run-root", required=True)
    plot_history.add_argument("--output", required=True)
    plot_history.add_argument("--transition-step", type=int)
    plot_history.add_argument("--rolling-steps", type=int, default=100)
    plot_history.set_defaults(func=command_plot_training_history)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

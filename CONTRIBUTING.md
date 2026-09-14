# Contributing

Thank you for helping improve the reproducibility of this project.

## Development workflow

1. Create a focused branch from `main`.
2. Do not edit an experiment YAML that has already produced a reported run.
   Copy it, assign a new `experiment_id`, and change one primary factor.
3. Keep data, checkpoints, maps, archives, secrets, and notebook outputs out of
   Git.
4. Add or update tests for changes to data handling, splits, models, losses,
   selection, or artifact contracts.
5. Run `pytest` and `xtseg validate-config` for every affected experiment.
6. Open a pull request describing the scientific question, controlled change,
   expected compute, and predeclared model-selection rule.

Results are comparable only when data, split, initialization, budget, and all
other controlled factors match. If more than one primary factor changes, label
the result exploratory rather than causal.

## Reproducibility metadata

Every reported run must preserve its Git commit, method hash, dataset ID, split
ID, seed, input checksums, initialization checksum, selected checkpoint role and
checksum, and validation/evaluation artifacts.

Do not use the independent verified points for hyperparameter tuning or
checkpoint selection.

## Adding components

Follow [`docs/EXTENDING.md`](docs/EXTENDING.md). Model and loss implementations
belong in the package, not in a notebook. A new component requires a versioned
config contract and numerical tests.

## Reporting security or data-governance issues

Do not open a public issue for a leaked credential, restricted dataset, or
sensitive location record. Follow [`SECURITY.md`](SECURITY.md).

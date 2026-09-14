# Reproducibility guide

## Public experiment matrix

| Config | Role | Initialization | Optimizer budget | Primary selection |
| --- | --- | --- | --- | --- |
| `pipeline_e0w_vertical.yaml` | Weighted-CE baseline | seed `20260910` | SGD, LR `1e-3`, momentum `0.9`, at most 3,120 steps | validation macro-F1, patience 6 |
| `pipeline_e2w_vertical_weighted_ce_to_dmi_loss_driven.yaml` | reported CE→DMI branch | locked CE checkpoint at step 2,652 | SGD, LR `3e-7`, 6,240 steps | fixed train-panel DMI loss |
| `pipeline_e2w_control_vertical_weighted_ce_to_dmi_lr1e-5.yaml` | matched no-morphology control | same locked CE checkpoint | SGD, LR `1e-5`, 6,240 steps | fixed train-panel DMI loss |
| `pipeline_e3m_vertical_weighted_ce_to_dmi_logit_additive_closing_lr1e-5.yaml` | learnable-morphology branch | U-Net submodule from the same CE checkpoint | SGD, LR `1e-5`, 6,240 steps | fixed train-panel DMI loss |

All rows use the same 2026 image contract, vertical split, frozen patch/batch
artifacts, batch size 16, and validation every 156 optimizer steps. The final two
rows are the one-factor-at-a-time comparison for morphology.

## 1. Create the environment

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[train,evaluate,report]"
```

Record `python --version`, `python -m pip freeze`, the full Git commit, GPU model,
CUDA version, and resolved experiment hash with every long run.

## 2. Verify and prepare data

```bash
EXPERIMENT=configs/experiments/pipeline_e0w_vertical.yaml
DATA_ROOT=/path/to/data
RUN_ROOT=/path/to/runs/weighted_ce

xtseg validate-config --experiment "$EXPERIMENT"
xtseg verify-data --experiment "$EXPERIMENT" --data-root "$DATA_ROOT"
xtseg run-pipeline \
  --experiment "$EXPERIMENT" \
  --data-root "$DATA_ROOT" \
  --run-root "$RUN_ROOT" \
  --through data
```

Prepared split, normalization, manifest, schedule, and validation-window files
are stored below `RUN_ROOT/prepared/` with their verification metadata.

## 3. Train or resume

```bash
xtseg run-pipeline \
  --experiment "$EXPERIMENT" \
  --data-root "$DATA_ROOT" \
  --run-root "$RUN_ROOT" \
  --through train
```

If a runtime disconnects, rerun the identical command. Resume is refused if the
Git commit, method hash, or run lock differs.

For a DMI continuation, provide the exact Weighted-CE model selected at step
2,652. The YAML locks its SHA-256 to
`ffce28223708a37848811d753313ab7c97fe332030b5f44fbf78571cf7a12311`:

```bash
xtseg run-pipeline \
  --experiment configs/experiments/pipeline_e2w_vertical_weighted_ce_to_dmi_loss_driven.yaml \
  --data-root "$DATA_ROOT" \
  --run-root /path/to/runs/ce_to_dmi \
  --initialization-checkpoint /path/to/E0w/FINAL_MODEL.pt \
  --through train
```

Use the matched-control or morphology YAML in the same command to reproduce
those branches. Never substitute another CE checkpoint without creating a new
experiment YAML and ID.

## 4. Generate the final map

```bash
xtseg run-pipeline \
  --experiment "$EXPERIMENT" \
  --data-root "$DATA_ROOT" \
  --run-root "$RUN_ROOT" \
  --through map
```

For checkpoint-initialized experiments, retain the same
`--initialization-checkpoint` argument. The map stage uses `FINAL_MODEL.pt`,
writes a georeferenced class-code GeoTIFF, and seals its checksum.

## 5. Perform independent evaluation once

Only after the protocol and model are frozen:

```bash
xtseg run-pipeline \
  --experiment "$EXPERIMENT" \
  --data-root "$DATA_ROOT" \
  --run-root "$RUN_ROOT" \
  --through evaluate \
  --confirm-independent-evaluation
```

Do not use the independent-point result to revise the same experiment. A new
hypothesis requires a new YAML and must be chosen using training/spatial
validation evidence only.

## 6. Audit the result

A reportable result must retain:

- source commit or release tag;
- resolved config and method hash;
- dataset and initialization-checkpoint SHA-256;
- `RUN_LOCK.json` and completion markers;
- validation history, per-class metrics, and training-step log;
- selected checkpoint role, step, and checksum;
- sealed map checksum; and
- independent-evaluation tables and checksum, if evaluation was authorized.

Use [`paper/experiments.yaml`](../paper/experiments.yaml) and
[`paper/results.yaml`](../paper/results.yaml) as the machine-readable publication
index rather than relying on notebook output.

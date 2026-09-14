# Extending the codebase

## New data

Add a new immutable YAML under `configs/datasets/`, including file roles,
acquisition/provenance metadata, grid expectations, band order, class map, and
SHA-256 values. Add a new experiment YAML that references it. Never edit an
existing paper data contract after a run.

## New spatial split

Add a versioned split YAML with the strategy, guard, patch policy, sampling seeds,
batch size, and leakage audit. A spatial block k-fold implementation already
exists in the package, but a paper fold must have its own reviewed config and
must not share pixels across train and validation inputs.

## New model

Implement an `nn.Module` factory under `src/xuanthuy_seg/models/`, expose it by a
model YAML using a fully qualified `factory`, and add tests for output shape,
forward/backward finiteness, checkpoint loading, and any claimed identity or
initialization behavior. Do not define a second U-Net inside a notebook.

## New loss

Implement the loss under `src/xuanthuy_seg/losses/`, register it in the loss
factory, and add numerical tests against an independently computed small example.
Specify input semantics (`logits` or probabilities), ignored-label behavior,
precision, stabilization, reduction, and diagnostics in its YAML.

## New experiment

Copy the nearest experiment YAML, give it a new filename and `experiment_id`, and
change one primary factor at a time. Validate the config and run the test suite
before committing:

```bash
xtseg validate-config --experiment configs/experiments/new_experiment.yaml
pytest
```

The pull request must state the research question, controlled difference,
expected GPU budget, selection rule fixed before training, and whether the
independent test remains sealed.

# Results provenance

This repository separates executable protocols from reported numerical results.
`paper/experiments.yaml` defines the role and status of every publication branch;
`paper/results.yaml` accepts only values copied from validated completion markers
or evaluation tables.

Before adding or changing a number:

1. verify the experiment ID and method hash against `resolved_config.json`;
2. verify `FINAL_MODEL.pt` and the sealed map against their completion markers;
3. record the exact checkpoint role and optimizer step;
4. identify whether the metric comes from spatial validation or independent
   verified points; and
5. include the source artifact path and SHA-256 in the manifest.

Validation metrics are eligible for development and checkpoint analysis.
Independent-point metrics are final outcomes and must not be used for tuning.
Fixed-train-panel DMI is an optimization monitor and must not be labeled as
validation loss or test loss.

The manifest deliberately uses `null` when an artifact-backed value has not yet
been imported. A missing value is preferable to a number reconstructed from a
plot or memory.

The matched `1e-5` comparison is generated with `xtseg compare-matched-runs`.
Its McNemar calculation is paired by verified-point identity and tests only the
discordant correct/incorrect outcomes. A non-significant result is not evidence
that the methods are equivalent. Connected-component, singleton, and boundary
counts describe map structure; they are not ground-truth accuracy metrics.

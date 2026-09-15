# Data access and integrity

## What is not distributed

This repository does not grant redistribution rights for the Sentinel-2 scene,
the historical label raster, the verified-point archive, trained checkpoints,
or derived GeoTIFF maps. Obtain the research data from the project authors or
their eventual archive, then place the files in one local `DATA_ROOT`.

The code never depends on a Google Drive or workstation-specific location.
Only the caller supplies `DATA_ROOT` and `RUN_ROOT`.

## Required files

The canonical data description is
[`configs/datasets/xuanthuy_may2026_historical_labels_unknown_date.yaml`](../configs/datasets/xuanthuy_may2026_historical_labels_unknown_date.yaml).
The current paper protocol expects:

| Role | File | SHA-256 |
| --- | --- | --- |
| Sentinel-2 image | `Sentinel2_2026-05-26_HSTDNNXUANTHUY_10bands_10m.tif` | `ba549aa47568537ebec8c7ffeaab7a993ec77d6e7f00a9816f8eef884bd2f3ee` |
| Historical labels | `rasterized_vector_categorized.tif` | `d02d44ac7d8249ed62ca1e4df6794ac9787accbc57ce73bf1df2c139217048d5` |
| Independent points | `verified_points_shapefiles.zip` | `3159aac0e6e8001a8c83d684c66c633575cdbcc3d72a9a3b0964473d14780c21` |

The image has 10 Sentinel-2 bands at a common 10 m grid in this order:
`B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12`. The expected grid is
1,374 rows by 1,821 columns in `EPSG:32648`.

Run the integrity check before requesting a GPU:

```bash
xtseg verify-data \
  --experiment configs/experiments/pipeline_e0w_vertical.yaml \
  --data-root /path/to/data
```

To inspect a candidate replacement scene without modifying a paper protocol:

```bash
xtseg inspect-inputs \
  --data-root /path/to/candidate-data \
  --image candidate_image.tif \
  --label candidate_labels.tif \
  --verified-points candidate_verified_points.zip
```

Create a new dataset YAML and a new experiment ID for any changed input. Never
replace a file while retaining its old checksum or reuse a completed run folder.

The four completed publication runs were executed before the label-time metadata
was corrected. Their experiment YAMLs therefore resolve through a verbatim
[execution snapshot](../configs/datasets/provenance/xuanthuy_may2026_historical_labels_execution_snapshot_v1.yaml)
that preserves the original `method_hash` values. The snapshot is provenance,
not the current factual description. The correction and its computational scope
are recorded in [`paper/metadata_corrections.yaml`](../paper/metadata_corrections.yaml).

## Label-time interpretation

The label raster is a historical land-cover reference. Its observation and
compilation time is unknown; the project does not assign it a reference date or
date precision. A previous execution contract recorded January 2026 without
sufficient evidence, and that metadata has been withdrawn. The raster bytes and
checksum did not change. Temporal mismatch is therefore a plausible source of
label noise, not an observed date difference or a measured noise transition
process.

## Independent evaluation

The 1,037 verified points are an independent final evaluation resource. They are
not used to construct patches, select a learning rate, set a stopping rule, or
choose a checkpoint. The CLI requires an explicit confirmation flag before the
archive is opened. Repeated model selection against those points would invalidate
their independent-test interpretation.

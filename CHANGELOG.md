# Changelog

All notable changes to the public research artifact are documented here.

## Unreleased

- Corrected the unsupported January 2026 label-time metadata: the historical
  label raster has unknown observation and compilation time.
- Added a canonical unknown-date dataset contract and retained the old parsed
  contract only as a clearly marked execution snapshot for artifact provenance.
- Added dataset-v2 validation that prevents an unknown label time from carrying
  a `reference_date` or `date_precision`.
- Completed the matched no-morphology DMI control provenance and result import.
- Added a reproducible matched-run comparison for validation histories, paired
  verified points, and final-map fragmentation.
- Added the artifact-backed E3m-vs-control interpretation and claim limits.

## 0.1.0 - 2026-09-15

- Initial public research-code release.
- Reproducible 10-band, 11-class U-Net pipeline with spatial leakage audit.
- Deterministic weighted CE and exact float64 DMI losses.
- CE-to-DMI checkpoint continuation with immutable lineage checks.
- Trainable depthwise 3x3 logit additive closing layer.
- Whole-scene GeoTIFF inference and gated independent-point evaluation.
- Paper experiment/result manifests, public Colab launcher, tests, and CI.

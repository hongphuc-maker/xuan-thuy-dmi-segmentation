# Xuan Thuy DMI Segmentation

Research code for reproducible 11-class semantic segmentation of a Sentinel-2
scene over Xuan Thuy. The repository supports the complete path from immutable
data contracts and leakage-audited spatial splits to training, map generation,
and a one-time independent point evaluation.

The paper experiments compare:

1. a U-Net trained with mildly weighted cross-entropy (Weighted CE);
2. the same U-Net continued from the locked Weighted-CE checkpoint with exact
   determinant mutual information (DMI) loss;
3. a matched DMI control at learning rate `1e-5`; and
4. the matched DMI branch with a trainable depthwise 3x3 logit additive
   closing layer.

The implementation is inspired by the original
[L_DMI paper](https://proceedings.neurips.cc/paper/2019/hash/8a1ee9f2b7abe6e88d1a479ab6a42c5e-Abstract.html),
but dense pixel batches, the spatial validation protocol, class weighting,
checkpoint selection, and learnable morphology are project adaptations. They
must not be attributed to the original paper.

## Repository scope

This public repository contains source code, tests, immutable YAML protocols,
and a Colab launcher. It intentionally does **not** contain Sentinel-2 rasters,
label rasters, verified points, trained weights, GeoTIFF predictions, or private
Drive paths. See [Data access](docs/DATA.md) for the expected files and checksums.

```text
configs/       immutable data, split, model, loss, and experiment contracts
src/           versioned Python package and `xtseg` command-line interface
tests/         unit and synthetic end-to-end tests
notebooks/     thin Colab launcher; no duplicate model or loss definitions
docs/          methods, data, reproducibility, and extension guides
paper/         machine-readable experiment and result manifests
```

## Install

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/hongphuc-maker/xuan-thuy-dmi-segmentation.git
cd xuan-thuy-dmi-segmentation
python -m pip install -e ".[train,evaluate,report]"
```

For tests:

```bash
python -m pip install -e ".[train,evaluate,test]"
pytest
```

## Reproduce an experiment

Choose an immutable experiment YAML and separate local directories for inputs
and outputs:

```bash
xtseg validate-config \
  --experiment configs/experiments/pipeline_e0w_vertical.yaml

xtseg verify-data \
  --experiment configs/experiments/pipeline_e0w_vertical.yaml \
  --data-root /path/to/data

xtseg run-pipeline \
  --experiment configs/experiments/pipeline_e0w_vertical.yaml \
  --data-root /path/to/data \
  --run-root /path/to/runs/weighted_ce \
  --through train
```

Re-running the same command resumes only when the saved run contract and the
resolved `method_hash` match. To create the sealed map after model selection,
change `--through train` to `--through map`. Independent verified points remain
locked until the explicit evaluation confirmation is supplied; they must never
be used to tune hyperparameters or select checkpoints.

The paper protocols are listed in [Reproducibility](docs/REPRODUCIBILITY.md).
For Colab, open [the public launcher](notebooks/01_public_colab_pipeline.ipynb).

## Reproducibility guarantees

- Data files are identified by semantic role and SHA-256, not by a personal path.
- The validation band is separated from training by a 48-pixel guard.
- Pixel-centred 224x224 patches and their batch schedule are frozen by seeds.
- Resolved data, split, model, loss, optimizer, selection, and stopping settings
  form a stable `method_hash`.
- Checkpoint continuation verifies the source checkpoint SHA-256.
- Whole-scene maps retain raster metadata and are sealed before independent
  point evaluation.
- CI runs unit tests plus a synthetic raster-to-map integration test.

## Results and citation

Only results tied to an immutable experiment, source revision, method hash, and
artifact checksum belong in `paper/results.yaml`. See
[Results provenance](docs/RESULTS.md); do not copy numbers from notebook output
by hand.

Artifact-audited independent-point results currently available are:

| Branch | OA | Macro-F1 over 7 verified classes | Status |
| --- | ---: | ---: | --- |
| Weighted CE | 0.920849 | 0.898755 | complete |
| Weighted CE to DMI, LR `3e-7` | 0.925676 | 0.909443 | complete |
| Weighted CE to DMI + logit closing, LR `1e-5` | 0.925676 | 0.909508 | complete |
| Matched no-morphology DMI control, LR `1e-5` | — | — | partial at step 3,432/6,240 |

These values cover 1,036 usable points from an input archive of 1,037 points;
only 7 of the 11 classes are represented in verified ground truth. The tiny
difference between the two completed DMI rows is descriptive, not evidence of a
morphology effect, because their learning rates differ. See the
[publication checklist](docs/PUBLICATION_CHECKLIST.md) for the matched-control
requirement.

Citation metadata is in [`CITATION.cff`](CITATION.cff). Article title, author
order, and DOI will be updated when the manuscript metadata is finalized.

## License

Source code is released under the [BSD 3-Clause License](LICENSE). Dataset,
labels, verification points, and model artifacts are separate research assets
and are not licensed by this repository.

# Methods implemented by this repository

## Spatial protocol

The primary split is a vertical validation band covering columns 1,122–1,388.
A 48-pixel guard is excluded on both sides. The split audit requires zero shared
input pixels between training and validation. This is stricter than assigning
overlapping random patches independently to train and validation.

Training uses 4,992 frozen, pixel-centred 224x224 patches. The manifest seed is
`20260905`; the batch-schedule seed is `20260906`; batch size is 16. Rare-class
streams for class codes 5, 6, and 7 receive 416 centres each, and DMI batches use
a minimum class-support rule of 128 pixels. At inference and validation, windows
use 112-pixel stride and overlapping logits are averaged before metrics or maps
are produced.

## Model

The baseline is a batch-normalized U-Net with 10 input channels, 11 output
logits, and 64 base channels. The morphology variant wraps the same U-Net with a
depthwise, class-specific, trainable 3x3 additive grayscale closing operator on
the logits.

For each class, the morphology layer performs a smooth dilation followed by a
smooth erosion using `logsumexp`, fixed temperature `beta=10`, replicate padding,
and a zero-initialized non-flat structuring element. Zero initialization preserves
the symmetric smooth-closing transform at hand-off and avoids an arbitrary class
bias. The output remains logits; the loss applies the required softmax exactly
once.

## Losses

Weighted CE uses deterministic masked cross-entropy. Class weights are the square
root of inverse unique train-core pixel frequency, clipped to `[0.5, 5.0]` and
computed from the prepared-data artifact. Ignored pixels have code 255.

Exact DMI is evaluated in float64. For valid pixels, the implementation computes
softmax probabilities and the 11x11 empirical joint matrix

```text
J = one_hot(target).T @ probability / N
```

then minimizes the negative log absolute determinant using `slogdet`, with no
matrix jitter. Rank, condition number, and minimum singular value are recorded as
diagnostics. A fixed panel of 11 frozen **training** batches provides a comparable
DMI optimization trajectory; it is not validation loss. Spatial validation DMI
is separately computed from the overlap-averaged validation mosaic.

## Checkpoint policy

- Weighted CE selects the best unrestricted spatial-validation macro-F1
  checkpoint. The configured 11-class acceptance gate is retained and its pass or
  failure must be reported.
- Historical CE-to-DMI uses a fixed 6,240-step DMI budget and selects its primary
  checkpoint by fixed-train-panel DMI loss.
- The matched `1e-5` control and morphology branch use the same DMI budget and
  primary rule, while also saving best validation macro-F1 and macro-IoU
  checkpoints.
- Independent verified points are evaluated only after model selection and map
  sealing.

## Relation to the original DMI paper

The DMI objective and the CE-to-DMI continuation idea follow Xu et al. (NeurIPS
2019). The original work studies classification under instance-independent label
noise. This project adapts that idea to dense, spatially correlated remote-sensing
segmentation with severe class imbalance.

The following are project-specific and are **not** claims from the DMI paper:

- spatial guard-band validation and pixel-centred patch construction;
- rare-class batch scheduling and 16x224x224 dense batches;
- mild class-weighted CE pretraining;
- learning rates, step budgets, monitoring panels, and checkpoint rules;
- the assumption that an unknown historical label time may induce temporally
  mismatched supervision; and
- learnable logit morphology.

The implementation therefore tests an adaptation of DMI; it does not constitute
a direct reproduction of the classification experiments or their noise model.

# Matched morphology comparison

## Controlled design

The no-morphology control and logit-closing branch use the same Sentinel-2 scene
acquired in 2026, the same historical label raster of unknown observation time,
the same vertical spatial split, frozen patch and batch artifacts,
Weighted-CE checkpoint, DMI loss, SGD settings, learning rate `1e-5`, 6,240-step
budget, validation cadence, seed, and primary checkpoint rule. The configured
depthwise 3x3 logit additive closing layer is the intended changed factor.

Both primary models were selected at DMI step 6,084 by minimum fixed-train-panel
DMI loss. That panel is a training monitor, not a validation or test set.

## Artifact-backed observations

| Outcome | No morphology | Logit closing | Closing − control |
| --- | ---: | ---: | ---: |
| Validation OA at selected step | 0.811708 | 0.813663 | +0.001955 |
| Validation macro-F1 at selected step | 0.453774 | 0.451592 | −0.002182 |
| Validation mIoU at selected step | 0.353072 | 0.351826 | −0.001246 |
| Fixed train-panel DMI loss | 39.013197 | 39.011818 | −0.001379 |
| Verified-point OA | 0.930502 | 0.925676 | −0.004826 |
| Verified-point macro-F1, 7 classes | 0.915419 | 0.909508 | −0.005911 |
| Final-map connected components | 2,075 | 1,508 | −27.3% |
| Final-map singleton components | 551 | 247 | −55.2% |
| Final-map boundary pixels | 130,376 | 124,283 | −4.67% |

Only 9 of 1,036 evaluated points changed correctness status: the control alone
was correct at 7 and morphology alone at 2. The two-sided exact McNemar value is
`p=0.1796875`. Eleven point predictions changed class in total. Class 9 (`Stx`)
had the largest verified F1 change, from 0.752809 to 0.716763 (−0.036046).

Across the common valid map footprint, 9,988 of 1,471,235 pixels (0.679%) changed
class. Map complexity is computed per class with 8-connected components and then
summed. Boundary pixels use valid 4-neighbour class differences.

## Supported interpretation

For this single seed, scene, checkpoint policy, and morphology definition, the
closing layer produced a materially less fragmented final map. It did not
improve the recorded validation macro metrics or independent-point accuracy over
the matched control. The pointwise exact test does not reject equal OA at the
0.05 level.

## Claims not supported

The experiment does not show that morphology improves accuracy. It also does
not prove equality or non-inferiority, because the verified set covers only 7 of
11 classes, the pointwise test does not model spatial dependence, and there are
no repeated seeds or independent scenes. Reduced fragmentation is a structural
map property and should not be called higher thematic accuracy without suitable
object- or boundary-level ground truth.

# Results: transition-exclusion follow-up

This 15-second transition-exclusion run was designed **after** the original results were inspected. It reuses the same subject splits; it is a sensitivity analysis, not fresh confirmation. The [original results](ORIGINAL_RESULTS.md) and `artifacts/original_results.json` remain available. The raw-sample forest is a post hoc matched-input comparison.

All 51 subjects and all 18 activity classes were used. The three fixed person-level splits hold out 10 subjects for testing each time. Scores below are arithmetic means and ranges across these overlapping splits, not independent confidence intervals.

| Training → test | Model | Macro-F1 mean (range) | Accuracy mean (range) | ECE mean | Coverage mean | Accepted accuracy mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| phone → phone | Prior only | 0.006 (0.006–0.006) | 0.056 (0.056–0.056) | 0.000 | 1.000 | 0.056 |
| phone → watch | Prior only | 0.006 (0.006–0.006) | 0.056 (0.056–0.056) | 0.000 | 1.000 | 0.056 |
| phone → phone | Logistic regression | 0.319 (0.308–0.329) | 0.332 (0.324–0.339) | 0.058 | 0.792 | 0.386 |
| phone → watch | Logistic regression | 0.240 (0.201–0.307) | 0.267 (0.231–0.333) | 0.127 | 0.928 | 0.274 |
| phone → phone | Feature forest | 0.315 (0.300–0.338) | 0.326 (0.306–0.350) | 0.129 | 0.834 | 0.360 |
| phone → watch | Feature forest | 0.263 (0.209–0.369) | 0.300 (0.241–0.401) | 0.129 | 0.743 | 0.356 |
| phone → phone | Raw-sample forest | 0.263 (0.251–0.269) | 0.282 (0.272–0.292) | 0.102 | 0.829 | 0.309 |
| phone → watch | Raw-sample forest | 0.140 (0.133–0.144) | 0.205 (0.204–0.207) | 0.087 | 0.762 | 0.255 |
| phone → phone | CNN | 0.266 (0.244–0.282) | 0.300 (0.274–0.320) | 0.060 | 0.792 | 0.358 |
| phone → watch | CNN | 0.193 (0.166–0.239) | 0.252 (0.214–0.301) | 0.117 | 0.850 | 0.283 |
| phone → phone | Patch transformer | 0.261 (0.247–0.274) | 0.293 (0.277–0.317) | 0.040 | 0.854 | 0.329 |
| phone → watch | Patch transformer | 0.169 (0.120–0.249) | 0.216 (0.172–0.292) | 0.165 | 0.938 | 0.223 |
| watch → watch | Watch RF fallback | 0.600 (0.573–0.620) | 0.605 (0.574–0.624) | 0.157 | 0.821 | 0.681 |

Macro-F1 weights the 18 activities equally. ECE is ten-bin expected calibration error (smaller is better). Coverage is the share of test windows the model accepts using a confidence cutoff set for 80% coverage on validation. Accepted accuracy only describes those kept windows; it should be read together with realized coverage.

## Original contrast, revisited on trimmed data

Transformer minus phone-trained feature forest macro-F1. The contrast was in the original protocol, but these amended data and repeated test people were not. Intervals resample the 10 held-out **people**, keeping each person's windows together.

| Split | Test device | Difference | Descriptive subject-bootstrap 95% interval |
| ---: | --- | ---: | --- |
| 11 | phone | -0.064 | [-0.097, -0.029] |
| 11 | watch | -0.090 | [-0.113, -0.067] |
| 23 | phone | -0.053 | [-0.089, -0.021] |
| 23 | watch | -0.071 | [-0.088, -0.053] |
| 37 | phone | -0.045 | [-0.070, -0.021] |
| 37 | watch | -0.120 | [-0.162, -0.076] |

The transformer was below the forest on both test devices in all three measured splits. The conclusion is limited to these trained models, features, people, activities, and recording conditions. It does not show that transformers cannot work for activity recognition.

## Matched input: transformer versus raw-sample forest

Both models see the same 200 ordered magnitude samples. The forest has 240 trees with the original fixed forest settings. This comparison was added after viewing the original test results, so its intervals are descriptive and do not constitute an untouched hypothesis test.

| Split | Test device | Transformer minus raw forest F1 | Descriptive subject-bootstrap 95% interval |
| ---: | --- | ---: | --- |
| 11 | phone | +0.004 | [-0.025, +0.037] |
| 11 | watch | -0.023 | [-0.048, -0.000] |
| 23 | phone | -0.019 | [-0.040, +0.003] |
| 23 | watch | -0.005 | [-0.018, +0.011] |
| 37 | phone | +0.010 | [-0.011, +0.032] |
| 37 | watch | +0.116 | [+0.086, +0.141] |

## Practical device finding

A forest trained on watch examples from the train people performs better on held-out watch examples than the forest trained on those people's phone examples. The mean watch macro-F1 values are 0.600 and 0.263, respectively. This changes both the training device and its placement, so it does not isolate one physical cause. A **post hoc**, paired subject analysis of this fallback appears in [AUDIT.md](AUDIT.md); it was not the predeclared model contrast.

## Important limitations

- A phone in a pocket is less informative for some fine hand activities than a watch on a wrist. Per-class F1 and confusion matrices are in [results.json](results.json) and the [report](index.html).
- The original feature forest uses hand-designed magnitude features; the CNN, transformer and added raw forest use the same magnitude samples. Matching input values does not equalize model tuning or inductive biases.
- The first 15 seconds were excluded based on a published data concern, after the original results were known. Retained windows shift in time and are not paired with the original windows.
- These are controlled scripted activities, not a field deployment or a medical device.
- Three test splits reuse subjects. Treat across-split ranges as a sensitivity check, not as independent trials.

See [PROTOCOL.md](PROTOCOL.md) for the frozen design and [AUDIT.md](AUDIT.md) for source rate and sample checks.

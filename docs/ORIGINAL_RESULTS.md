# Results

All 51 subjects and all 18 activity classes were used. The three fixed person-level splits hold out 10 subjects for testing each time. Scores below are arithmetic means and ranges across these overlapping splits, not independent confidence intervals.

| Training → test | Model | Macro-F1 mean (range) | Accuracy mean (range) | ECE mean | Coverage mean | Accepted accuracy mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| phone → phone | Prior only | 0.006 (0.006–0.006) | 0.056 (0.056–0.056) | 0.000 | 1.000 | 0.056 |
| phone → watch | Prior only | 0.006 (0.006–0.006) | 0.056 (0.055–0.056) | 0.001 | 1.000 | 0.056 |
| phone → phone | Logistic regression | 0.318 (0.309–0.326) | 0.328 (0.322–0.334) | 0.049 | 0.790 | 0.380 |
| phone → watch | Logistic regression | 0.246 (0.199–0.314) | 0.273 (0.229–0.344) | 0.119 | 0.928 | 0.282 |
| phone → phone | Random forest | 0.324 (0.311–0.344) | 0.335 (0.315–0.358) | 0.135 | 0.828 | 0.375 |
| phone → watch | Random forest | 0.267 (0.215–0.367) | 0.305 (0.249–0.400) | 0.123 | 0.757 | 0.359 |
| phone → phone | CNN | 0.265 (0.248–0.276) | 0.301 (0.277–0.328) | 0.053 | 0.782 | 0.364 |
| phone → watch | CNN | 0.192 (0.165–0.244) | 0.248 (0.207–0.308) | 0.149 | 0.932 | 0.252 |
| phone → phone | Patch transformer | 0.261 (0.245–0.270) | 0.301 (0.290–0.318) | 0.041 | 0.777 | 0.359 |
| phone → watch | Patch transformer | 0.185 (0.142–0.271) | 0.237 (0.190–0.319) | 0.149 | 0.935 | 0.246 |
| watch → watch | Watch RF fallback | 0.599 (0.572–0.622) | 0.602 (0.575–0.623) | 0.150 | 0.822 | 0.677 |

Macro-F1 weights the 18 activities equally. ECE is ten-bin expected calibration error (smaller is better). Coverage is the share of test windows the model accepts using a confidence cutoff set for 80% coverage on validation. Accepted accuracy only describes those kept windows; it should be read together with realized coverage.

## Predeclared transformer comparison

Transformer minus phone-trained forest macro-F1. Intervals resample the 10 held-out **people**, keeping each person's windows together.

| Split | Test device | Difference | Descriptive subject-bootstrap 95% interval |
| ---: | --- | ---: | --- |
| 11 | phone | -0.075 | [-0.107, -0.040] |
| 11 | watch | -0.073 | [-0.089, -0.056] |
| 23 | phone | -0.066 | [-0.093, -0.044] |
| 23 | watch | -0.076 | [-0.099, -0.053] |
| 37 | phone | -0.048 | [-0.077, -0.024] |
| 37 | watch | -0.096 | [-0.126, -0.064] |

The transformer was below the forest on both test devices in all three measured splits. The conclusion is limited to these trained models, features, people, activities, and recording conditions. It does not show that transformers cannot work for activity recognition.

## Practical device finding

A forest trained on watch examples from the train people performs better on held-out watch examples than the forest trained on those people's phone examples. The mean watch macro-F1 values are 0.599 and 0.267, respectively. This changes both the training device and its placement, so it does not isolate one physical cause. A **post hoc**, paired subject analysis of this fallback appears in [AUDIT.md](AUDIT.md); it was not the predeclared model contrast.

## Important limitations

- A phone in a pocket is less informative for some fine hand activities than a watch on a wrist. Per-class F1 and confusion matrices are in [results.json](results.json) and the [report](index.html).
- The forest uses hand-designed magnitude features; the CNN and transformer use the magnitude sequence. Model architecture is not the only difference.
- These are controlled scripted activities, not a field deployment or a medical device.
- Three test splits reuse subjects. Treat across-split ranges as a sensitivity check, not as independent trials.

See [PROTOCOL.md](PROTOCOL.md) for the frozen design and [AUDIT.md](AUDIT.md) for source rate and sample checks.

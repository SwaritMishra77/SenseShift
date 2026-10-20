# Data and result audit

## Source and preprocessing

Official UCI WISDM Smartphone and Smartwatch Activity and Biometrics, ID 507. The original archive is verified by SHA256 before extracting only accelerometer streams.

The rWISDM analysis (Heydarian and Doyle, 2023, https://arxiv.org/abs/2305.10222) describes inconsistent phone axes, varying sampling rates, and initial activity transitions. We use vector magnitude, invariant to physical axis rotations and signed permutations, rather than their axis-translation heuristic. The existing timestamp resampling handles the rate finding. The amended run removes the first 15 seconds of each continuous activity segment **before** filtering. A synthetic test checks that transitions cannot bleed into kept samples. The XYZ orientation was not inferred or corrected participant by participant; magnitude cannot restore directional cues.

The repair paper tests five activities. Applying its 15-second exclusion to all 18 activities here is an exploratory sensitivity choice, not proof that every discarded interval is mislabeled.

- Original archive SHA256: `6e0147f181d8f5275918db65eeb9dc842a0185d37cb6b2f30cbf74ee6e205469`
- All 51 participants: 29,694 ten-second windows; phone 14,848, watch 14,846.
- Original run: 32,055 windows; amended run: 29,694 windows. Retained windows start 15 seconds later per activity segment and are not paired with original windows.
- All 18 activity labels remain present on each device.
- Every original extracted subject file has an individual SHA256 in the local `data/processed/windows.json` manifest (raw and cache are excluded from Git).

| Device | Rows | Segments | Windows | Subject median native rates |
| --- | ---: | ---: | ---: | --- |
| phone | 4,804,403 | 928 | 14,848 | 20 Hz: 27, 25 Hz: 9, 35 Hz: 3, 50 Hz: 11, 51 Hz: 1 |
| watch | 3,777,046 | 958 | 14,846 | 20 Hz: 47, 50 Hz: 4 |

The source description gives 20 Hz as a nominal rate, but the raw timestamps vary substantially by device and participant. Resampling by timestamp avoids silently rejecting the faster streams. The phone and watch clocks have different origins, so this project makes no paired-window claim.

## Chronology and sensitivity

The original three-seed study was evaluated first. The transition exclusion and raw-sample forest were added only after viewing its test results. The same test people appear in both runs. The table compares scores on different retained windows and is descriptive, not a paired gain or fresh test.

| Model / test | Original mean macro-F1 | Amended mean macro-F1 |
| --- | ---: | ---: |
| rf / phone | 0.324 | 0.315 |
| rf / watch | 0.267 | 0.263 |
| transformer / phone | 0.261 | 0.261 |
| transformer / watch | 0.185 | 0.169 |
| watch_rf_fallback / watch | 0.599 | 0.600 |

A score change cannot be attributed solely to removing transition noise: the retained windows have shifted boundaries and the models were retrained on a different sample.

## Held-out cohort checks

| Seed | Phone train / validation / test windows | Watch train / validation / test windows |
| ---: | --- | --- |
| 11 | 9,136 / 2,864 / 2,848 | 9,132 / 2,875 / 2,839 |
| 23 | 8,880 / 2,832 / 3,136 | 8,867 / 2,832 / 3,147 |
| 37 | 8,848 / 3,120 / 2,880 | 8,828 / 3,140 / 2,878 |

## Post hoc fallback comparison

A watch-specific forest uses watch training participants. The comparison below reuses the same held-out watch windows as the phone forest, with a paired participant bootstrap. This is **exploratory**; only transformer versus phone forest was the predeclared contrast.

| Seed | Watch-specific minus phone-trained macro-F1 | Descriptive 95% interval |
| ---: | ---: | --- |
| 11 | +0.363 | [+0.248, +0.441] |
| 23 | +0.398 | [+0.284, +0.461] |
| 37 | +0.251 | [+0.216, +0.290] |

## Interpretation limits

- Windows from the same participant are correlated; a window-level standard error would be misleading.
- The three participant splits overlap, so their scores are sensitivity checks, not independent replications.
- The original feature forest sees designed magnitude statistics; the raw-sample forest and deep models see the same 200 magnitude samples. This reduces input differences but does not isolate architecture from training choices.
- Follow-up comparisons reuse previously viewed test people; subject-bootstrap intervals reflect sampling variation within these held-out subjects, not fresh replication.
- Device placement, hardware, and native sampling patterns change together. A phone-to-watch drop cannot be attributed to a single physical cause.
- Source-device calibration is not automatically valid on a watch. The reported realized test coverage and expected calibration error expose this failure mode.
- These controlled scripted activities are not a real-world deployment benchmark.

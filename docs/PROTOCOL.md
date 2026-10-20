# Experiment protocol

The original protocol was specified before examining model results. The project tests
**whether an activity classifier trained on phone motion remains useful on
people it has never seen, and what changes if only a watch is available**.

## Cohort and input

- WISDM Smartphone and Smartwatch Activity and Biometrics (UCI ID 507).
- Original raw **accelerometer** recordings from all 51 subjects and all 18
  activity codes. Gyroscope recordings are outside the present scope.
- Break each stream at a new activity, nonincreasing timestamp, or gap over
  300 ms. Estimate each segment's native rate from timestamp differences.
  Low-pass streams sampled above 22 Hz to 8 Hz, then interpolate x/y/z onto a
  20 Hz clock. The original run began at segment onset. The 2026-10-19
  sensitivity run excludes the first 15 seconds of every continuous activity
  segment **before filtering**, following the concern about activity transitions
  in Heydarian and Doyle's rWISDM paper. Their study evaluated five activities;
  applying this rule to all 18 here is a sensitivity choice, not a proven
  label repair. Form ten-second, 200-point
  **nonoverlapping** windows. Never
  bridge an activity boundary or a clock gap.
- Classical models receive 17 magnitude statistics and frequency features;
  deep models receive 200-point magnitude traces. This is orientation-invariant,
  but it discards directional clues. No features contain subject ID, timestamp,
  device identifier, or activity code.
- The two device clocks have different origins. Phone and watch windows are
  independent samples; they must **not** be described as synchronized pairs.

## Recorded amendment and input comparison

After the original three-seed results were inspected, we reviewed Heydarian
and Doyle, *rWISDM: Repaired WISDM* (2023),
https://arxiv.org/abs/2305.10222. The original source ZIP already handled
varying sample rates with timestamps. The predictor uses vector magnitude,
which is unchanged by signed axis permutations or physical device rotations;
therefore we do not perform an XYZ axis-repair heuristic that would change the
vector norm. Their concern about the first seconds of activity recordings
motivated a separate 15-second transition-exclusion sensitivity run. This was
**not** in the original preregistered-like protocol, and it reuses the same
people in the original train/validation/test split. Retained windows can start
at different times, so scores across the two runs are not paired window scores.

The revised run also adds `rf_raw`: the same 240-tree forest trained on the
same ordered 200-sample acceleration-magnitude vector given to the CNN and
transformer. No handcrafted features are passed to it. This compares models
on the same sensor values, but architectures still impose different inductive
biases, training methods, and tuning. The matched-input contrast is post hoc
and descriptive; it is not a new untouched test. Exact earlier results are
preserved in `artifacts/original_results.json` and `docs/ORIGINAL_RESULTS.md`.

## Split and training

Subject IDs are shuffled separately for fixed seeds **11, 23, 37**, yielding
31 train, 10 validation, and 10 test subjects each time. All recordings from a
subject stay together across both devices. Repeated splits share subjects, so
they are sensitivity analyses rather than independent trials.

Train the prior-only dummy, standardized multinomial logistic regression,
240-tree random forest, small three-layer CNN, and two-layer patch transformer
on **phone train** only. Select a deep checkpoint on phone validation macro-F1
using at most 12 epochs and patience 3. The forest and other hyperparameters
are fixed before test evaluation. Fit probability temperature and select the
threshold for 80% source validation coverage using **phone validation** only.
The same frozen models and threshold evaluate **phone test** and **watch test**.

Separately, train a watch-specific random forest using watch train and fit
its temperature/threshold on watch validation. Compare it to direct transfer
when a phone sensor is unavailable. This is a sensor-specific fallback, not a
paired or imputed missing-sensor fusion model.

## Reported outcomes

Primary outcome: 18-class macro-F1 for phone test and watch test. Also report
accuracy, balanced accuracy, expected calibration error (10 bins), negative
log-likelihood, realized coverage and accuracy among accepted predictions at
the threshold selected on validation. Show per-class support/F1 and the most
frequent confusions. Raw performance on held-out people, rather than window
random splits, is the meaningful test.

**Predeclared comparison:** transformer minus random-forest macro-F1 on each
test device. A 2,000-resample paired bootstrap over the **10 test subjects**
gives a descriptive 95% interval. It is not a p-value or proof of superiority;
windows within one person are correlated. Repeat all models across the three
fixed splits to inspect sensitivity. In particular, a better source-device
score does not establish a device-transfer gain.

## Limits known before the run

The recordings are controlled activities from a small cohort; they do not
measure real deployment in arbitrary homes or wearable hardware. Device
location is confounded with phone versus watch. Activity-specific missing
segments and person-specific recording rates must be audited. Temperature
calibration on the source device may fail after a device shift. Results from
the test sets should not be used to redesign hyperparameters and retest on
the same splits.

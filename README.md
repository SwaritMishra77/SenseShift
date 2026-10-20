# SenseShift

**Can an activity recognizer trained on a phone handle a new person or a watch?**

SenseShift studies that question on the [WISDM smartphone and smartwatch dataset](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset). It uses all 51 participants and all 18 activities, with person-level train, validation, and test splits. The code runs a prior baseline, logistic regression, a feature-based forest, a raw-sample forest, a small CNN, and a small patch transformer. A separate forest trained on the watch is tested as a fallback when the phone is unavailable.

The project includes complete Python source, tests, the original and amended protocols, measured results and held-out predictions from both runs, a data-quality audit, and a [standalone results page](docs/index.html). The raw recordings are fetched from UCI rather than copied into the repository.

## Main finding

In the original experiment, the phone-trained feature forest averaged **0.324 macro-F1 on phone** and **0.267 on watch**; the transformer averaged **0.261** and **0.185**. After excluding the first 15 seconds of each activity segment and rerunning with the same PyTorch version, the forest averaged **0.315 / 0.263** and the transformer **0.261 / 0.169**. Removing transition windows did not reliably improve performance. The added raw-sample forest averaged **0.263 / 0.140** with the same 200 input values as the transformer; their relative ranking varies by split and device. The watch-trained fallback averaged **0.600 on watch** in the amended run. Transition trimming and the matched-input forest were added **after** the original results were inspected and reuse the same test people; they are exploratory, not fresh confirmation. See [the results](docs/RESULTS.md), [the original results](docs/ORIGINAL_RESULTS.md), and [the audit](docs/AUDIT.md).

These are results on controlled, scripted activities. They do not establish how the models would perform in unrestricted daily life.

## Reproduce

Use Python 3.11 or 3.12, several GB of free disk space, and enough time for the full 51-person experiment. The official download is about 296 MB; extracted sensor streams, caches, and dependencies need more space.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e '.[deep]'
senseshift acquire --download
senseshift prepare
senseshift run
senseshift audit
senseshift report
senseshift export-models
python -m unittest discover -s tests -v
```

The default `prepare` excludes the first 15 seconds of every activity segment before filtering. To reproduce the untrimmed original run without overwriting the amended outputs, use `senseshift prepare --transition-seconds 0 --cache data/processed/original_windows.npz`, then `senseshift run --cache data/processed/original_windows.npz --output artifacts/original_reproduction/results.json --models dummy linear rf cnn transformer`. The supplied `artifacts/original_results.json` documents that earlier run. The download command verifies the pinned SHA256 of the original archive and extracts only phone/watch accelerometer files. If you already downloaded the official ZIP, run `senseshift acquire --archive path/to/the-official.zip`. The exact outer ZIP used for this release has SHA256 `6e0147f181d8f5275918db65eeb9dc842a0185d37cb6b2f30cbf74ee6e205469`. If UCI replaces the file, the verifier stops rather than silently using different data.

Run the report by opening `docs/index.html` locally. It is self-contained and also works from GitHub Pages when the Pages source is set to the repository's `docs/` directory. The HTML is generated from `artifacts/results.json`, not hand-entered example numbers.

To replay an actual held-out watch window after preparing the data, run `senseshift replay --model watch_rf_fallback --device watch`. Try `--model rf` on that same window to compare direct phone-to-watch transfer. The command uses the included seed-11 checkpoints and prints the true activity, prediction, confidence, and whether the fixed validation cutoff accepts it. Use `--subject` and `--window` to inspect another seed-11 test person's window.

For the versions used in the amended run, see [requirements-tested.txt](requirements-tested.txt) and the `runtime` field in `artifacts/results.json`. The CPU build of PyTorch can be installed from its official wheel index; other PyTorch variants can produce different deep-model results. Installing the optional deep dependencies can take time. `senseshift run --models dummy linear rf rf_raw` runs just the classical models, but is a **different** experiment and will not reproduce the deep-model contrasts in the committed report.

## How the study works

1. Each activity stream is separated at label changes or recording gaps. Some raw phone streams run at roughly 25 or 50 Hz despite the nominal 20 Hz description. The amended run removes 15 transition seconds before filtering faster streams and resampling by timestamp to 20 Hz. This amendment followed the [rWISDM analysis](https://arxiv.org/abs/2305.10222); the original run is preserved.
2. It forms nonoverlapping, ten-second windows from acceleration magnitude. Each window belongs to one person, one device, and one activity. Phone and watch windows are not assumed to have synchronized clocks.
3. For each of three fixed seeds, 31 people are used for training, ten for validation, and ten for testing. A person's recordings never cross these groups, including across devices.
4. The six phone models train on phone recordings only. Model selection, calibration, and a confidence cutoff targeting 80% coverage use phone validation only. Tests use held-out people's phones and watches. The watch fallback trains and validates separately on watch recordings from the same train/validation people.
5. Macro-F1, balanced accuracy, accuracy, calibration error, realized coverage, and accuracy among accepted predictions are recorded for every test. A participant-level bootstrap summarizes the predeclared transformer-versus-forest difference.

See the [fixed protocol](docs/PROTOCOL.md), [data and result audit](docs/AUDIT.md), and [dataset notes](docs/DATA.md) for exact rules and limitations.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/senseshift/data.py` | Official archive verification, parsing, transition exclusion, resampling, feature extraction |
| `src/senseshift/deep.py` | CNN and patch transformer training |
| `src/senseshift/experiment.py` | Subject splits, model comparisons and saved predictions |
| `src/senseshift/evaluation.py` | Calibration, selective prediction, metrics, participant bootstrap |
| `src/senseshift/audit.py` | Data integrity and exploratory fallback comparison |
| `src/senseshift/export.py` | Verified seed-11 forest checkpoint export |
| `src/senseshift/inference.py` | Replay an actual held-out example from saved models |
| `src/senseshift/report.py` | Self-contained HTML report generated from actual results |
| `tests/` | Windowing and leakage-sensitive protocol tests |
| `artifacts/results.json` / `artifacts/original_results.json` | Measured amended and original three-split runs |
| `artifacts/heldout_predictions.csv.gz` / `artifacts/original_heldout_predictions.csv.gz` | Per-window predictions for the amended and original runs |
| `docs/` | Protocol, audit, results, and the local/GitHub Pages report |
| `scripts/package_release.py` | Checks for missing source and creates a GitHub-ready ZIP |

`data/raw/` and `data/processed/` are intentionally ignored by Git. The original data belongs to its creators, and the processed windows can be rebuilt exactly from the verified archive. The small deep checkpoints and verified seed-11 forests in `artifacts/models/` are included along with the training code and measured predictions. Only load serialized models from a source you trust.

## Reading the results correctly

- The 18 classes include similar seated/hand activities that are hard to distinguish from a phone in a pocket. A watch may contain different information because it is worn on the wrist. Device and placement change together here; the experiment cannot identify one cause in isolation.
- The feature forest uses designed magnitude features. The raw forest and deep models receive the same magnitude sequence, but this does not equalize their training and inductive biases.
- The added data trim and raw forest are post hoc analyses on already viewed test subjects. A score change from trimming also changes which windows exist and where they start.
- The three test splits overlap in subjects. Intervals resample the ten test **people**, not thousands of correlated windows, and are descriptive rather than a formal claim of general superiority.
- Do not choose new hyperparameters by looking at the saved test results and then call a rerun on those same people a fresh test.

## Data credit

Gary Weiss, *WISDM Smartphone and Smartwatch Activity and Biometrics Dataset*, UCI Machine Learning Repository, 2019, [DOI:10.24432/C5HK59](https://doi.org/10.24432/C5HK59). Dataset license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Project code: MIT license.

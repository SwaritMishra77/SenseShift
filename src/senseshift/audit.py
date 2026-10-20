"""Prepare human-readable data and result audits from immutable outputs."""

import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ACTIVITIES, LABELS
from .evaluation import subject_bootstrap_difference


def write_audits(meta_file=Path("data/processed/windows.json"),
                 results_file=Path("artifacts/results.json"),
                 original_results_file=Path("artifacts/original_results.json"),
                 predictions_file=Path("artifacts/heldout_predictions.csv.gz"),
                 docs_dir=Path("docs")):
    meta = json.loads(Path(meta_file).read_text())
    results = json.loads(Path(results_file).read_text())
    original = json.loads(Path(original_results_file).read_text())
    if results["data_sha256"] != meta["cache_sha256"] or results["data_sha256"] == original["data_sha256"]:
        raise ValueError("The amended run must match the new cache, distinct from the original")
    if [r["subjects"] for r in results["runs"]] != [r["subjects"] for r in original["runs"]]:
        raise ValueError("Original and amended subject splits must be identical")
    docs_dir = Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(predictions_file, "rt") as f:
        preds = pd.read_csv(f)
    audits = []
    for run in results["runs"]:
        seed = run["seed"]
        direct = preds[(preds.seed == seed) & (preds.device == "watch") & (preds.model == "rf")].sort_values("window_id")
        fallback = preds[(preds.seed == seed) & (preds.device == "watch") & (preds.model == "watch_rf_fallback")].sort_values("window_id")
        if len(direct) != len(fallback):
            raise ValueError(f"Missing fallback window predictions for seed {seed}")
        if not np.array_equal(direct[["window_id", "subject", "true_activity"]].to_numpy(),
                              fallback[["window_id", "subject", "true_activity"]].to_numpy()):
            raise ValueError(f"Predictions not aligned by watch window for seed {seed}")
        label_to_i = {v: i for i, v in enumerate(LABELS)}
        y = direct.true_activity.map(label_to_i).to_numpy()
        onehot = lambda x: np.eye(len(LABELS))[x.map(label_to_i).to_numpy()]
        contrast = subject_bootstrap_difference(y, onehot(fallback.predicted_activity),
                                                onehot(direct.predicted_activity),
                                                direct.subject.to_numpy(), seed=seed + 777)
        audits.append({"seed": seed, "watch_specific_minus_transferred_rf":
                       contrast["difference_transformer_minus_rf"],
                       "subject_bootstrap_95_interval": contrast["subject_bootstrap_95_interval"]})
    (docs_dir / "fallback_analysis.json").write_text(json.dumps(audits, indent=2) + "\n")
    summary = ["# Data and result audit", "", "## Source and preprocessing", "",
               "Official UCI WISDM Smartphone and Smartwatch Activity and Biometrics, ID 507. "
               "The original archive is verified by SHA256 before extracting only accelerometer streams.", "",
               "The rWISDM analysis (Heydarian and Doyle, 2023, "
               "https://arxiv.org/abs/2305.10222) describes inconsistent phone axes, "
               "varying sampling rates, and initial activity transitions. We use vector "
               "magnitude, invariant to physical axis rotations and signed permutations, "
               "rather than their axis-translation heuristic. The existing timestamp "
               "resampling handles the rate finding. The amended run removes the first "
               f"{meta['transition_seconds']:g} seconds of each continuous activity segment "
               "**before** filtering. A synthetic test checks that transitions cannot "
               "bleed into kept samples. The XYZ orientation was not inferred or corrected "
               "participant by participant; magnitude cannot restore directional cues.", "",
               "The repair paper tests five activities. Applying its 15-second exclusion "
               "to all 18 activities here is an exploratory sensitivity choice, not "
               "proof that every discarded interval is mislabeled.", "",
               f"- Original archive SHA256: `{meta['source_archive_sha256']}`",
               f"- All 51 participants: {meta['windows']:,} ten-second windows; "
               f"phone {meta['by_device']['phone']:,}, watch {meta['by_device']['watch']:,}.",
               f"- Original run: {original['dataset']['windows']:,} windows; "
               f"amended run: {meta['windows']:,} windows. Retained windows start 15 seconds "
               "later per activity segment and are not paired with original windows.",
               "- All 18 activity labels remain present on each device.",
               "- Every original extracted subject file has an individual SHA256 in the local "
               "`data/processed/windows.json` manifest (raw and cache are excluded from Git).", "",
               "| Device | Rows | Segments | Windows | Subject median native rates |", "| --- | ---: | ---: | ---: | --- |"]
    for device in ("phone", "watch"):
        files = meta["per_file"][device]
        counts = Counter(round(x["median_native_rate_hz"]) for x in files)
        rates = ", ".join(f"{hz} Hz: {n}" for hz, n in sorted(counts.items()))
        summary.append(f"| {device} | {sum(x['raw_rows'] for x in files):,} | "
                       f"{sum(x['segments'] for x in files):,} | "
                       f"{sum(x['windows'] for x in files):,} | {rates} |")
    summary += ["", "The source description gives 20 Hz as a nominal rate, but the raw "
                "timestamps vary substantially by device and participant. Resampling by "
                "timestamp avoids silently rejecting the faster streams. The phone and "
                "watch clocks have different origins, so this project makes no paired-window claim.", "",
                "## Chronology and sensitivity", "",
                "The original three-seed study was evaluated first. The transition exclusion "
                "and raw-sample forest were added only after viewing its test results. "
                "The same test people appear in both runs. The table compares scores on "
                "different retained windows and is descriptive, not a paired gain or fresh test.", "",
                "| Model / test | Original mean macro-F1 | Amended mean macro-F1 |",
                "| --- | ---: | ---: |"]
    for model, device in (("rf", "phone"), ("rf", "watch"),
                          ("transformer", "phone"), ("transformer", "watch"),
                          ("watch_rf_fallback", "watch")):
        previous = np.mean([r["models"][model]["tests"][device]["macro_f1"]
                            for r in original["runs"]])
        updated = np.mean([r["models"][model]["tests"][device]["macro_f1"]
                           for r in results["runs"]])
        summary.append(f"| {model} / {device} | {previous:.3f} | {updated:.3f} |")
    summary += ["", "A score change cannot be attributed solely to removing "
                "transition noise: the retained windows have shifted boundaries and "
                "the models were retrained on a different sample.", "",
                "## Held-out cohort checks", "",
                "| Seed | Phone train / validation / test windows | Watch train / validation / test windows |",
                "| ---: | --- | --- |"]
    for r in results["runs"]:
        split = r["subjects"]
        if set(split["train"]) & set(split["test"] + split["validation"]) or set(split["test"]) & set(split["validation"]):
            raise ValueError("Subject overlap in saved results")
        counts = r["counts"]
        summary.append(f"| {r['seed']} | "
                       f"{counts['phone_train']:,} / {counts['phone_validation']:,} / {counts['phone_test']:,} | "
                       f"{counts['watch_train']:,} / {counts['watch_validation']:,} / {counts['watch_test']:,} |")
    summary += ["", "## Post hoc fallback comparison", "",
                "A watch-specific forest uses watch training participants. The comparison "
                "below reuses the same held-out watch windows as the phone forest, with a "
                "paired participant bootstrap. This is **exploratory**; only transformer "
                "versus phone forest was the predeclared contrast.", "",
                "| Seed | Watch-specific minus phone-trained macro-F1 | Descriptive 95% interval |",
                "| ---: | ---: | --- |"]
    for r in audits:
        l, u = r["subject_bootstrap_95_interval"]
        summary.append(f"| {r['seed']} | {r['watch_specific_minus_transferred_rf']:+.3f} | "
                       f"[{l:+.3f}, {u:+.3f}] |")
    summary += ["", "## Interpretation limits", "",
                "- Windows from the same participant are correlated; a window-level standard error would be misleading.",
                "- The three participant splits overlap, so their scores are sensitivity checks, not independent replications.",
                "- The original feature forest sees designed magnitude statistics; the raw-sample "
                "forest and deep models see the same 200 magnitude samples. This reduces input "
                "differences but does not isolate architecture from training choices.",
                "- Follow-up comparisons reuse previously viewed test people; subject-bootstrap "
                "intervals reflect sampling variation within these held-out subjects, not fresh replication.",
                "- Device placement, hardware, and native sampling patterns change together. A phone-to-watch "
                "drop cannot be attributed to a single physical cause.",
                "- Source-device calibration is not automatically valid on a watch. The reported realized "
                "test coverage and expected calibration error expose this failure mode.",
                "- These controlled scripted activities are not a real-world deployment benchmark.", ""]
    (docs_dir / "AUDIT.md").write_text("\n".join(summary))
    return audits

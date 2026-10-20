"""Locked experiment: source phone models, held-out subjects, shifted watch tests."""

from __future__ import annotations

import gzip
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import ACTIVITIES, FEATURE_NAMES, LABELS, sha256
from .deep import fit_deep, predict_deep, save_deep
from .evaluation import (confidence_cutoff, fit_temperature, metrics,
                         split_subjects, subject_bootstrap_difference,
                         subset, temperature_transform)

DEFAULT_SEEDS = (11, 23, 37)
DEFAULT_MODELS = ("dummy", "linear", "rf", "rf_raw", "cnn", "transformer")


def _class_index(code):
    return np.searchsorted(np.asarray(LABELS), code).astype(np.int64)


def _diagnostics(y, prob):
    pred = prob.argmax(1)
    cm = confusion_matrix(y, pred, labels=np.arange(len(LABELS)))
    support = cm.sum(axis=1)
    per_class_f1 = f1_score(y, pred, average=None, labels=np.arange(len(LABELS)), zero_division=0)
    confusions = sorted(((int(cm[i, j]), LABELS[i], LABELS[j])
                         for i in range(len(LABELS)) for j in range(len(LABELS)) if i != j), reverse=True)
    return {"confusion_matrix": cm.tolist(),
            "class_support": dict(zip(LABELS, support.astype(int).tolist())),
            "class_f1": dict(zip(LABELS, [float(x) for x in per_class_f1])),
            "top_confusions": [{"count": n, "true": a, "predicted": b}
                               for n, a, b in confusions[:8]]}


def _runtime_versions():
    result = {"python": sys.version.split()[0]}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "torch"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def run(cache: Path, output: Path, *, seeds=DEFAULT_SEEDS,
        models=DEFAULT_MODELS, max_epochs: int = 12, save_models: bool = False):
    cache, output = Path(cache), Path(output)
    meta = json.loads(cache.with_suffix(".json").read_text())
    if sha256(cache) != meta["cache_sha256"]:
        raise ValueError("Window cache checksum does not match its manifest")
    with np.load(cache, allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    subjects = data["subject"]
    labels = _class_index(data["label"])
    if len(set(subjects)) != 51 or len(set(data["label"])) != len(LABELS):
        raise ValueError("Incomplete cohort or missing activity")
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction_file = output.parent / "heldout_predictions.csv.gz"
    results = {"project": "SenseShift", "data_sha256": meta["cache_sha256"],
               "source_archive_sha256": meta["source_archive_sha256"],
               "runtime": _runtime_versions(),
               "protocol": {"train_device": "phone", "target_device": "watch",
                            "transition_seconds": meta["transition_seconds"],
                            "subject_split": "31 train / 10 validation / 10 test per seed",
                            "seeds": list(seeds), "window_samples": 200, "overlap": 0,
                            "source_validation_coverage_target": 0.8,
                            "model_selection": "best source-validation macro-F1 for deep models",
                            "temperature_and_threshold": "fit only on phone validation",
                            "primary_contrast": "original transformer minus feature forest, revisited after transition exclusion",
                            "secondary_contrast": "transformer minus raw-sample forest (same 200 input magnitudes; post hoc)"},
               "dataset": {"subjects": meta["subjects"], "windows": meta["windows"],
                           "by_device": meta["by_device"], "by_activity": meta["by_activity"]},
               "runs": []}
    with gzip.open(prediction_file, "wt", encoding="utf-8") as csv:
        csv.write("seed,device,window_id,subject,true_activity,model,predicted_activity,confidence,correct,accepted\n")
        for seed in seeds:
            split = split_subjects(subjects, seed)
            indexes = {f"{device}_{group}": subset(data, device, split[group])
                       for device in ("phone", "watch")
                       for group in ("train", "validation", "test")}
            tr, val = indexes["phone_train"], indexes["phone_validation"]
            test = {d: indexes[f"{d}_test"] for d in ("phone", "watch")}
            for key, idx in indexes.items():
                if len(idx) == 0 or len(set(labels[idx])) != len(LABELS):
                    raise ValueError(f"Seed {seed} has incomplete activity coverage: {key}")
            run_result = {"seed": seed, "subjects": split,
                          "counts": {k: len(v) for k, v in indexes.items()},
                          "models": {}, "hypothesis": {}}
            stored_probs = {}
            for name in models:
                if name in ("dummy", "linear", "rf", "rf_raw"):
                    if name == "dummy":
                        model = DummyClassifier(strategy="prior")
                    elif name == "linear":
                        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
                    else:
                        model = RandomForestClassifier(n_estimators=240, min_samples_leaf=2,
                                                       max_features="sqrt", n_jobs=4,
                                                       random_state=seed)
                    inputs = data["sequence"] if name == "rf_raw" else data["features"]
                    model.fit(inputs[tr], labels[tr])
                    if not np.array_equal(model.classes_ if name == "dummy" else
                                          model[-1].classes_ if name == "linear" else model.classes_,
                                          np.arange(len(LABELS))):
                        raise ValueError("Model missing a class")
                    predict = lambda ix: model.predict_proba(inputs[ix])
                    additional = {}
                    if name == "rf":
                        ranks = np.argsort(model.feature_importances_)[::-1][:8]
                        additional["feature_importance"] = [
                            {"feature": FEATURE_NAMES[j], "importance": float(model.feature_importances_[j])}
                            for j in ranks]
                else:
                    model, deep_meta = fit_deep(name, data["sequence"][tr], labels[tr],
                                                data["sequence"][val], labels[val], seed=seed,
                                                max_epochs=max_epochs)
                    predict = lambda ix: _logit_to_proba(predict_deep(
                        model, data["sequence"][ix],
                        mean=deep_meta["mean"], std=deep_meta["std"]))
                    additional = {"training": deep_meta}
                    if save_models:
                        destination = output.parent / "models"
                        destination.mkdir(exist_ok=True)
                        save_deep(model, destination / f"{name}_seed{seed}.pt", deep_meta)
                val_prob = predict(val)
                temp = fit_temperature(val_prob, labels[val])
                cutoff = confidence_cutoff(temperature_transform(val_prob, temp))
                result = {"temperature": temp, "validation_confidence_cutoff": cutoff,
                          "validation": metrics(labels[val], temperature_transform(val_prob, temp), cutoff),
                          **additional, "tests": {}}
                for device, ix in test.items():
                    prob = temperature_transform(predict(ix), temp)
                    stored_probs[(name, device)] = prob
                    result["tests"][device] = {**metrics(labels[ix], prob, cutoff),
                                                **_diagnostics(labels[ix], prob)}
                    for j, p in zip(ix, prob):
                        cls = int(np.argmax(p))
                        csv.write(f"{seed},{device},{j},{subjects[j]},{data['label'][j]},{name},"
                                  f"{LABELS[cls]},{float(np.max(p)):.6f},{int(cls==labels[j])},"
                                  f"{int(np.max(p)>=cutoff)}\n")
                run_result["models"][name] = result
                print(f"seed {seed:>2} {name:<12} source F1 {result['tests']['phone']['macro_f1']:.3f} "
                      f"shifted F1 {result['tests']['watch']['macro_f1']:.3f}", flush=True)
            if "rf" in models:
                # Separately trained fallback answers the practical missing-phone question.
                watch_rf = RandomForestClassifier(n_estimators=240, min_samples_leaf=2,
                                                  max_features="sqrt", n_jobs=4, random_state=seed)
                watch_train = indexes["watch_train"]
                watch_val = indexes["watch_validation"]
                watch_rf.fit(data["features"][watch_train], labels[watch_train])
                val_prob = watch_rf.predict_proba(data["features"][watch_val])
                temp = fit_temperature(val_prob, labels[watch_val])
                cutoff = confidence_cutoff(temperature_transform(val_prob, temp))
                prob = temperature_transform(watch_rf.predict_proba(data["features"][test["watch"]]), temp)
                run_result["models"]["watch_rf_fallback"] = {
                    "temperature": temp, "validation_confidence_cutoff": cutoff,
                    "validation": metrics(labels[watch_val], temperature_transform(val_prob, temp), cutoff),
                    "tests": {"watch": {**metrics(labels[test["watch"]], prob, cutoff),
                                         **_diagnostics(labels[test["watch"]], prob)}}}
                for j, p in zip(test["watch"], prob):
                    cls = int(np.argmax(p))
                    csv.write(f"{seed},watch,{j},{subjects[j]},{data['label'][j]},watch_rf_fallback,"
                              f"{LABELS[cls]},{float(np.max(p)):.6f},{int(cls==labels[j])},"
                              f"{int(np.max(p)>=cutoff)}\n")
                print(f"seed {seed:>2} watch RF fallback shifted F1 "
                      f"{run_result['models']['watch_rf_fallback']['tests']['watch']['macro_f1']:.3f}", flush=True)
            if "rf" in models and "transformer" in models:
                for device, ix in test.items():
                    run_result["hypothesis"][device] = subject_bootstrap_difference(
                        labels[ix], stored_probs[("transformer", device)], stored_probs[("rf", device)],
                        subjects[ix], seed=seed + 423)
            if "rf_raw" in models and "transformer" in models:
                run_result["matched_input_contrast"] = {}
                for device, ix in test.items():
                    run_result["matched_input_contrast"][device] = subject_bootstrap_difference(
                        labels[ix], stored_probs[("transformer", device)], stored_probs[("rf_raw", device)],
                        subjects[ix], seed=seed + 624)
            results["runs"].append(run_result)
            output.write_text(json.dumps(results, indent=2) + "\n")  # checkpoint after each seed
    results["predictions_file"] = prediction_file.name
    output.write_text(json.dumps(results, indent=2) + "\n")
    return results


def _logit_to_proba(logits):
    from scipy.special import softmax
    return softmax(logits, axis=1)

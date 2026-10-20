"""Subject-held-out evaluation and subject-level uncertainty summaries."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, log_loss)

from .data import LABELS


def split_subjects(all_subjects, seed: int, n_val: int = 10, n_test: int = 10) -> dict:
    ids = np.sort(np.unique(all_subjects))
    if len(ids) != 51:
        raise ValueError(f"Expected 51 subjects, got {len(ids)}")
    shuffled = np.random.default_rng(seed).permutation(ids)
    split = {"train": sorted(shuffled[n_val + n_test:].tolist()),
             "validation": sorted(shuffled[n_test:n_val + n_test].tolist()),
             "test": sorted(shuffled[:n_test].tolist())}
    if set(split["train"]) & set(split["validation"]) or set(split["test"]) & set(split["train"] + split["validation"]):
        raise AssertionError("Subject leakage")
    return split


def subset(dataset, device: str, ids):
    return np.flatnonzero((dataset["device"] == device) & np.isin(dataset["subject"], ids))


def fit_temperature(prob: np.ndarray, labels: np.ndarray) -> float:
    """Fit one scalar temperature on source validation predictions only."""
    logits = np.log(np.clip(prob, 1e-12, 1.0))
    fn = lambda t: log_loss(labels, softmax(logits / t, axis=1), labels=np.arange(len(LABELS)))
    optimum = minimize_scalar(fn, bounds=(0.3, 5.0), method="bounded")
    return float(optimum.x)


def temperature_transform(prob: np.ndarray, t: float) -> np.ndarray:
    return softmax(np.log(np.clip(prob, 1e-12, 1.0)) / t, axis=1)


def confidence_cutoff(prob: np.ndarray, coverage: float = 0.8) -> float:
    """Choose threshold on source validation; evaluation never tunes it."""
    confidence = np.max(prob, axis=1)
    return float(np.quantile(confidence, 1 - coverage, method="lower"))


def expected_calibration_error(labels, prob, bins: int = 10) -> float:
    conf, pred = prob.max(axis=1), prob.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        mask = (conf >= edges[i]) & (conf < edges[i + 1] if i < bins - 1 else conf <= 1)
        if mask.any():
            total += mask.mean() * abs((pred[mask] == labels[mask]).mean() - conf[mask].mean())
    return float(total)


def metrics(labels, prob, cutoff: float) -> dict:
    labels = np.asarray(labels)
    pred = prob.argmax(axis=1)
    mask = prob.max(axis=1) >= cutoff
    accuracy = float(accuracy_score(labels[mask], pred[mask])) if mask.any() else None
    return {"n": len(labels), "accuracy": float(accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, average="macro", labels=np.arange(len(LABELS)), zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
            "ece_10": expected_calibration_error(labels, prob),
            "nll": float(log_loss(labels, prob, labels=np.arange(len(LABELS)))),
            "coverage": float(mask.mean()), "accepted_accuracy": accuracy,
            "accepted_count": int(mask.sum())}


def subject_bootstrap_difference(y, p_a, p_b, subjects, *, seed: int = 421,
                                 repeats: int = 2000) -> dict:
    """Paired by held-out subject; windows are dependent within each subject."""
    ids = np.unique(subjects)
    rng = np.random.default_rng(seed)
    pred_a, pred_b = p_a.argmax(1), p_b.argmax(1)
    def score(ix):
        return f1_score(y[ix], pred_a[ix], average="macro", labels=np.arange(len(LABELS)), zero_division=0) - \
               f1_score(y[ix], pred_b[ix], average="macro", labels=np.arange(len(LABELS)), zero_division=0)
    observed = float(score(np.arange(len(y))))
    by_subject = {s: np.flatnonzero(subjects == s) for s in ids}
    values = np.empty(repeats)
    for i in range(repeats):
        sample = rng.choice(ids, size=len(ids), replace=True)
        index = np.concatenate([by_subject[s] for s in sample])
        values[i] = score(index)
    return {"difference_transformer_minus_rf": observed,
            "subject_bootstrap_95_interval": [float(x) for x in np.quantile(values, [0.025, 0.975])],
            "resamples": repeats, "subjects": len(ids)}

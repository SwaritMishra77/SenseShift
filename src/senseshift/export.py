"""Export small, verified seed-11 models for a reproducible local example."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from .data import LABELS
from .evaluation import split_subjects, subset


def export_forests(cache=Path("data/processed/windows.npz"),
                   result_file=Path("artifacts/results.json"),
                   out=Path("artifacts/models")):
    with np.load(cache, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    results = json.loads(Path(result_file).read_text())
    seed = 11
    saved = next(r for r in results["runs"] if r["seed"] == seed)
    split = split_subjects(data["subject"], seed)
    if split != saved["subjects"]:
        raise ValueError("Saved result participant split does not match cache")
    labels = np.searchsorted(np.asarray(LABELS), data["label"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for device, name in (("phone", "rf"), ("phone", "rf_raw"),
                         ("watch", "watch_rf_fallback")):
        train = subset(data, device, split["train"])
        test = subset(data, device, split["test"])
        inputs = data["sequence"] if name == "rf_raw" else data["features"]
        model = RandomForestClassifier(n_estimators=240, min_samples_leaf=2,
                                       max_features="sqrt", n_jobs=4, random_state=seed)
        model.fit(inputs[train], labels[train])
        measured = f1_score(labels[test], model.predict(inputs[test]),
                            average="macro", labels=np.arange(len(LABELS)), zero_division=0)
        recorded = saved["models"][name]["tests"][device]["macro_f1"]
        if not np.isclose(measured, recorded, atol=1e-12):
            raise ValueError(f"Exported model disagrees with measured run: {name}")
        path = out / f"{name}_seed{seed}.joblib"
        joblib.dump(model, path, compress=3)
        outputs[name] = {"path": str(path), "bytes": path.stat().st_size,
                         "verified_macro_f1": measured}
    (out / "README.md").write_text(
        "# Saved models\n\n"
        "These three seed-11 forests are checked against the measured test macro-F1 "
        "before export. The six small `.pt` files contain the trained CNN "
        "and transformer checkpoints for all three seeds.\n\n"
        "Only load `.joblib` or `.pt` files from a source you trust. "
        "The raw dataset and window cache are downloaded/built locally.\n")
    return outputs

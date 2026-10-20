"""Replay one held-out ten-second window using a saved seed-11 checkpoint."""

import json
from pathlib import Path

import joblib
import numpy as np

from .data import ACTIVITIES, LABELS
from .deep import make_model, predict_deep
from .evaluation import subset, temperature_transform


def replay(model_name: str = "watch_rf_fallback", device: str = "watch",
           subject: int | None = None, window: int = 0,
           cache=Path("data/processed/windows.npz"),
           result_file=Path("artifacts/results.json"),
           checkpoints=Path("artifacts/models")) -> dict:
    if model_name not in ("rf", "rf_raw", "watch_rf_fallback", "cnn", "transformer"):
        raise ValueError("Choose rf, rf_raw, watch_rf_fallback, cnn, or transformer")
    if device not in ("phone", "watch"):
        raise ValueError("Device must be phone or watch")
    if model_name == "watch_rf_fallback" and device != "watch":
        raise ValueError("The watch fallback is trained for watch recordings")
    with np.load(cache, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    result = json.loads(Path(result_file).read_text())
    run = next(r for r in result["runs"] if r["seed"] == 11)
    test_subjects = run["subjects"]["test"]
    subject = test_subjects[0] if subject is None else subject
    if subject not in test_subjects:
        raise ValueError(f"Subject must belong to the seed-11 test group: {test_subjects}")
    ix = subset(data, device, [subject])
    if not 0 <= window < len(ix):
        raise IndexError(f"Window index {window} outside the {len(ix)} available windows")
    j = int(ix[window])
    checkpoints = Path(checkpoints)
    if model_name in ("rf", "rf_raw", "watch_rf_fallback"):
        # Serialized pickle formats must only be loaded from trusted project files.
        model = joblib.load(checkpoints / f"{model_name}_seed11.joblib")
        inputs = data["sequence"] if model_name == "rf_raw" else data["features"]
        prob = model.predict_proba(inputs[j:j + 1])
    else:
        import torch
        saved = torch.load(checkpoints / f"{model_name}_seed11.pt", map_location="cpu",
                           weights_only=True)
        model = make_model(model_name, len(LABELS))
        model.load_state_dict(saved["state_dict"])
        logits = predict_deep(model, data["sequence"][j:j + 1],
                              mean=saved["metadata"]["mean"], std=saved["metadata"]["std"])
        from scipy.special import softmax
        prob = softmax(logits, axis=1)
    settings = run["models"][model_name]
    calibrated = temperature_transform(prob, settings["temperature"])[0]
    predicted = LABELS[int(calibrated.argmax())]
    confidence = float(calibrated.max())
    actual = str(data["label"][j])
    return {"seed": 11, "device": device, "subject": int(subject),
            "window_index_in_subject": window, "window_id_in_cache": j,
            "model": model_name, "true_activity": ACTIVITIES[actual],
            "predicted_activity": ACTIVITIES[predicted],
            "confidence": confidence,
            "validation_cutoff": settings["validation_confidence_cutoff"],
            "accepted": confidence >= settings["validation_confidence_cutoff"],
            "correct": predicted == actual,
            "note": "Replay of a labeled, held-out WISDM window; not a live wearable application"}

"""Official WISDM v2 acquisition and deterministic, leakage-safe windowing."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/507/"
    "wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip"
)
SOURCE_SHA256 = "6e0147f181d8f5275918db65eeb9dc842a0185d37cb6b2f30cbf74ee6e205469"
ACTIVITIES = {
    "A": "walking", "B": "jogging", "C": "stairs", "D": "sitting",
    "E": "standing", "F": "typing", "G": "teeth", "H": "soup",
    "I": "chips", "J": "pasta", "K": "drinking", "L": "sandwich",
    "M": "kicking", "O": "catch", "P": "dribbling", "Q": "writing",
    "R": "clapping", "S": "folding",
}
LABELS = tuple(ACTIVITIES)
WINDOW = 200
MAX_GAP_NS = 300_000_000
TRANSITION_SECONDS = 15.0
FEATURE_NAMES = (
    "mean", "std", "median", "min", "max", "q10", "q90", "iqr",
    "rms", "mean_abs_jerk", "std_jerk", "peak_hz", "spectral_entropy",
    "power_low", "power_mid", "power_high", "autocorr_1s",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def acquire(outer_zip: Path, raw_dir: Path, *, download: bool = False) -> dict:
    """Extract only original phone/watch accelerometer files, never rehost raw data."""
    outer_zip = Path(outer_zip)
    raw_dir = Path(raw_dir)
    if download and not outer_zip.exists():
        outer_zip.parent.mkdir(parents=True, exist_ok=True)
        temp = outer_zip.with_suffix(".part")
        try:
            with urllib.request.urlopen(SOURCE_URL, timeout=120) as src, temp.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            temp.replace(outer_zip)
        finally:
            temp.unlink(missing_ok=True)
    if not outer_zip.exists():
        raise FileNotFoundError(f"Provide {outer_zip} or use --download")
    actual_hash = sha256(outer_zip)
    if actual_hash != SOURCE_SHA256:
        raise ValueError(f"Unexpected archive SHA256 {actual_hash}; expected {SOURCE_SHA256}")
    with ZipFile(outer_zip) as outer:
        nested = outer.read("wisdm-dataset.zip")
    from io import BytesIO
    found = set()
    raw_hashes = {}
    with ZipFile(BytesIO(nested)) as z:
        for info in z.infolist():
            parts = Path(info.filename).parts
            if (len(parts) != 5 or parts[:2] != ("wisdm-dataset", "raw")
                    or parts[2] not in ("phone", "watch") or parts[3] != "accel"
                    or not parts[4].startswith("data_") or not parts[4].endswith(".txt")):
                continue
            device = parts[2]
            output = raw_dir / device / parts[4]
            output.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, output.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            found.add((device, parts[4]))
            raw_hashes[f"{device}/{parts[4]}"] = sha256(output)
    if len(found) != 102 or len({p[1].split("_")[1] for p in found}) != 51:
        raise ValueError(f"Expected 51 files per device; extracted {len(found)}")
    manifest = {"dataset": "UCI WISDM smartphone/watch (ID 507)",
                "source_url": SOURCE_URL, "archive_sha256": actual_hash,
                "devices": ["phone", "watch"], "sensor": "accelerometer",
                "files": len(found), "raw_sha256": raw_hashes}
    (raw_dir / "source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def features(mag: np.ndarray) -> np.ndarray:
    """Orientation-invariant features from a 10 s acceleration magnitude trace."""
    x = np.asarray(mag, dtype=np.float64)
    centered = x - x.mean()
    power = np.abs(np.fft.rfft(centered)) ** 2
    freq = np.fft.rfftfreq(len(x), d=0.05)
    power[0] = 0.0
    p = power / max(power.sum(), 1e-12)
    spec_entropy = -np.sum(p * np.log(p + 1e-12)) / np.log(len(p))
    dx = np.diff(x)
    lag = 20
    autocorr = float(np.corrcoef(centered[:-lag], centered[lag:])[0, 1]) if x.std() > 1e-6 else 0.0
    if not np.isfinite(autocorr):
        autocorr = 0.0
    vals = [x.mean(), x.std(), np.median(x), x.min(), x.max(),
            *np.quantile(x, [0.1, 0.9]), np.quantile(x, 0.75) - np.quantile(x, 0.25),
            np.sqrt(np.mean(x * x)), np.mean(np.abs(dx)), dx.std(),
            freq[np.argmax(power)], spec_entropy,
            p[(freq >= 0.1) & (freq < 0.8)].sum(),
            p[(freq >= 0.8) & (freq < 2.0)].sum(),
            p[(freq >= 2.0) & (freq <= 5.0)].sum(), autocorr]
    return np.asarray(vals, dtype=np.float32)


def windows_from_file(path: Path, *, strict: bool = True,
                      transition_seconds: float = TRANSITION_SECONDS) -> tuple[list, dict]:
    """Segment by activity/time, then resample physical seconds to 20 Hz."""
    if transition_seconds < 0:
        raise ValueError("transition_seconds must be nonnegative")
    # pandas parses the semicolon on the final coordinate as a string.
    df = pd.read_csv(path, header=None, names=["subject", "activity", "time", "x", "y", "z"],
                     dtype={"subject": "int32", "activity": "str", "time": "int64",
                            "x": "float32", "y": "float32", "z": "str"},
                     on_bad_lines="skip")
    df["z"] = pd.to_numeric(df["z"].str.rstrip(";"), errors="coerce")
    subject = int(path.name.split("_")[1])
    raw_count = len(df)
    valid = (df["subject"].eq(subject) & df["activity"].isin(LABELS)
             & df[["x", "y", "z"]].notna().all(axis=1))
    if strict and not valid.all():
        raise ValueError(f"Invalid records in {path}: {(~valid).sum()}")
    df = df[valid].reset_index(drop=True)
    time = df["time"].to_numpy(dtype=np.int64)
    labels = df["activity"].to_numpy(dtype="U1")
    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    if len(df) < WINDOW:
        return [], {"raw_rows": raw_count, "valid_rows": len(df), "windows": 0,
                    "segments": 0, "transition_seconds": transition_seconds,
                    "transition_segments": 0}
    delta = np.diff(time)
    cuts = np.r_[0, np.flatnonzero((labels[1:] != labels[:-1]) |
                                    (delta <= 0) | (delta > MAX_GAP_NS)) + 1, len(df)]
    output = []
    original_rates = []
    discarded_transition_segments = 0
    for segment_id, (start, end) in enumerate(zip(cuts[:-1], cuts[1:])):
        if end - start < WINDOW // 2:
            continue
        t = (time[start:end] - time[start]).astype(np.float64) / 1e9
        if t[-1] < transition_seconds + (WINDOW - 1) / 20:
            continue
        native_rate = 1 / float(np.median(np.diff(t)))
        original_rates.append(native_rate)
        # Trim before the zero-phase low-pass filter: filtering the discarded
        # transition would smear its values into the first retained window.
        kept = int(np.searchsorted(t, transition_seconds, side="left"))
        signal = xyz[start + kept:end].astype(np.float64)
        support = t[kept:]
        # Suppress >8 Hz components before reducing streams recorded at 25/50 Hz.
        # Device rates are estimated within each continuous activity segment.
        if native_rate > 22 and len(signal) > 30:
            sos = butter(4, 8, fs=native_rate, output="sos")
            signal = sosfiltfilt(sos, signal, axis=0)
        starts = np.arange(transition_seconds,
                           t[-1] - (WINDOW - 1) / 20 + 1e-8, 10.0)
        discarded_transition_segments += int(transition_seconds > 0)
        for offset in starts:
            target = offset + np.arange(WINDOW) / 20.0
            resampled = np.stack([np.interp(target, support, signal[:, axis]) for axis in range(3)], axis=1)
            mag = np.linalg.norm(resampled, axis=1).astype(np.float32)
            origin = int(time[start] + round(offset * 1e9))
            output.append((subject, labels[start], segment_id, origin, mag, features(mag)))
    return output, {"raw_rows": raw_count, "valid_rows": len(df),
                    "windows": len(output), "segments": len(cuts) - 1,
                    "transition_seconds": transition_seconds,
                    "transition_segments": discarded_transition_segments,
                    "median_native_rate_hz": float(np.median(original_rates)) if original_rates else None}


def build(raw_dir: Path, cache: Path, *, transition_seconds: float = TRANSITION_SECONDS) -> dict:
    raw_dir, cache = Path(raw_dir), Path(cache)
    source = raw_dir / "source.json"
    if not source.exists() or json.loads(source.read_text()).get("archive_sha256") != SOURCE_SHA256:
        raise ValueError("Raw data lacks the verified UCI acquisition manifest; run acquire")
    source_manifest = json.loads(source.read_text())
    paths = {d: sorted((raw_dir / d).glob("data_*_accel_*.txt")) for d in ("phone", "watch")}
    if any(len(v) != 51 for v in paths.values()):
        raise ValueError("Need all 51 original accelerometer files for each device; run acquire")
    stats, records = {}, []
    for device, files in paths.items():
        stats[device] = []
        for path in files:
            file_hash = sha256(path)
            if source_manifest.get("raw_sha256", {}).get(f"{device}/{path.name}") != file_hash:
                raise ValueError(f"Raw file differs from verified archive: {path}")
            rows, details = windows_from_file(path, transition_seconds=transition_seconds)
            records.extend((device, *row) for row in rows)
            stats[device].append({"file": path.name, "sha256": file_hash, **details})
    device, subject, label, segment, timestamp, mag, feat = zip(*records)
    np.savez_compressed(cache, device=np.asarray(device, dtype="U5"),
                        subject=np.asarray(subject, dtype=np.int16),
                        label=np.asarray(label, dtype="U1"),
                        segment=np.asarray(segment, dtype=np.int32),
                        timestamp=np.asarray(timestamp, dtype=np.int64),
                        sequence=np.asarray(mag, dtype=np.float32),
                        features=np.asarray(feat, dtype=np.float32))
    meta = {"window_samples": WINDOW, "sampling_rate_hz": 20, "stride_samples": WINDOW,
            "transition_seconds": transition_seconds,
            "feature_names": FEATURE_NAMES, "labels": ACTIVITIES,
            "subjects": len(set(subject)), "windows": len(records),
            "by_device": dict(Counter(device)), "by_activity": dict(Counter(label)),
            "source_archive_sha256": SOURCE_SHA256, "per_file": stats,
            "cache_sha256": sha256(cache)}
    cache.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta

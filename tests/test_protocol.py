import tempfile
import unittest
from pathlib import Path

import numpy as np

from senseshift.data import LABELS, features, windows_from_file
from senseshift.evaluation import (confidence_cutoff, split_subjects,
                                   subject_bootstrap_difference)


class WindowingTests(unittest.TestCase):
    def test_time_based_resampling_at_both_native_rates(self):
        outputs = []
        for hz in (20, 50):
            with self.subTest(native_hz=hz), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "data_1600_accel_phone.txt"
                with path.open("w") as f:
                    for activity, start in (("A", 0.0), ("B", 35.0)):
                        for i in range(21 * hz):
                            t = i / hz
                            f.write(f"1600,{activity},{int((start+t)*1e9)},"
                                    f"{9.8+np.sin(2*np.pi*t)},0,0;\n")
                windows, audit = windows_from_file(path, transition_seconds=0)
                self.assertEqual([w[1] for w in windows], ["A", "A", "B", "B"])
                self.assertEqual(len(windows), 4)
                self.assertEqual(audit["segments"], 2)
                self.assertTrue(all(w[4].shape == (200,) for w in windows))
                outputs.append(windows[0][5])
        self.assertLess(abs(float(outputs[0][11] - outputs[1][11])), 0.11)  # 1 Hz peak

    def test_clock_gap_splits_an_activity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data_1600_accel_watch.txt"
            with path.open("w") as f:
                for start in (0, 40):
                    for i in range(20 * 12):
                        f.write(f"1600,A,{int((start+i/20)*1e9)},9,0,0;\n")
            windows, audit = windows_from_file(path, transition_seconds=0)
            self.assertEqual(len(windows), 2)
            self.assertEqual(audit["segments"], 2)

    def test_transition_exclusion_uses_timestamp_seconds(self):
        for hz in (20, 50):
            with self.subTest(native_hz=hz), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "data_1600_accel_phone.txt"
                with path.open("w") as f:
                    for i in range(35 * hz):
                        t = i / hz
                        # Artificial transition is confined to the first 15 seconds.
                        amp = 20 if t < 15 else 9
                        f.write(f"1600,A,{int(t*1e9)},{amp},0,0;\n")
                windows, audit = windows_from_file(path)
                self.assertEqual(len(windows), 2)
                self.assertEqual(audit["transition_seconds"], 15)
                self.assertTrue(all(w[3] >= 15_000_000_000 for w in windows))
                self.assertTrue(all(np.allclose(w[4], 9, atol=0.05) for w in windows))

    def test_magnitude_invariant_to_axis_permutation_and_sign(self):
        t = np.linspace(0, 10, 200, endpoint=False)
        xyz = np.stack([np.sin(t), np.cos(2*t), 9.8 + np.sin(3*t)], axis=1)
        a = np.linalg.norm(xyz, axis=1)
        b = np.linalg.norm(xyz[:, [2, 0, 1]] * [-1, 1, -1], axis=1)
        np.testing.assert_allclose(a, b)
        np.testing.assert_allclose(features(a), features(b))


class EvaluationTests(unittest.TestCase):
    def test_holdout_people_never_overlap(self):
        ids = np.arange(1600, 1651)
        x = split_subjects(ids, 11)
        self.assertEqual([len(x[k]) for k in ("train", "validation", "test")], [31, 10, 10])
        self.assertEqual(len(set(x["train"] + x["validation"] + x["test"])), 51)

    def test_bootstrap_resamples_people_and_identity_is_zero(self):
        y = np.tile(np.arange(len(LABELS)), 2)
        p = np.eye(len(LABELS))[y]
        subjects = np.repeat([1600, 1601], len(LABELS))
        result = subject_bootstrap_difference(y, p, p, subjects, repeats=50)
        self.assertEqual(result["difference_transformer_minus_rf"], 0.0)
        self.assertEqual(result["subject_bootstrap_95_interval"], [0.0, 0.0])

    def test_validation_confidence_threshold(self):
        confidence = np.linspace(0.1, 0.99, 100)
        proba = np.c_[confidence, 1 - confidence]
        cut = confidence_cutoff(proba, coverage=0.8)
        self.assertGreaterEqual(float(np.mean(proba.max(1) >= cut)), 0.8)


if __name__ == "__main__":
    unittest.main()

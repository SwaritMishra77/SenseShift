"""Check that the published numbers can be reconstructed from held-out predictions."""

import csv
import gzip
import json
import unittest
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import f1_score

from senseshift.data import LABELS


ROOT = Path(__file__).resolve().parents[1]


class PublishedArtifactTests(unittest.TestCase):
    def test_both_runs_match_their_heldout_predictions(self):
        for filename in ("results.json", "original_results.json"):
            with self.subTest(run=filename):
                result = json.loads((ROOT / "artifacts" / filename).read_text())
                groups = defaultdict(lambda: ([], [], []))
                with gzip.open(ROOT / "artifacts" / result["predictions_file"], "rt") as file:
                    for row in csv.DictReader(file):
                        key = (int(row["seed"]), row["model"], row["device"])
                        y, pred, people = groups[key]
                        y.append(row["true_activity"])
                        pred.append(row["predicted_activity"])
                        people.append(int(row["subject"]))
                for run in result["runs"]:
                    test_people = set(run["subjects"]["test"])
                    for name, model in run["models"].items():
                        for device, measurements in model["tests"].items():
                            y, pred, people = groups.pop((run["seed"], name, device))
                            self.assertTrue(set(people) <= test_people)
                            self.assertEqual(len(y), measurements["n"])
                            measured = f1_score(y, pred, average="macro",
                                                labels=list(LABELS), zero_division=0)
                            self.assertAlmostEqual(measured, measurements["macro_f1"], places=12)
                self.assertFalse(groups, "Unexpected extra prediction groups")


if __name__ == "__main__":
    unittest.main()

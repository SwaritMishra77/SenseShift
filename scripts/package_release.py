"""Validate a self-contained GitHub source ZIP without bundling UCI raw data."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import json

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md", "LICENSE", "pyproject.toml", "requirements-tested.txt",
    "src/senseshift/__init__.py", "src/senseshift/cli.py", "src/senseshift/data.py",
    "src/senseshift/deep.py", "src/senseshift/experiment.py", "src/senseshift/evaluation.py",
    "src/senseshift/audit.py", "src/senseshift/report.py", "src/senseshift/export.py",
    "src/senseshift/inference.py", "tests/test_protocol.py", "tests/test_artifacts.py",
    "docs/PROTOCOL.md", "docs/DATA.md", "docs/RESULTS.md", "docs/ORIGINAL_RESULTS.md",
    "docs/AUDIT.md",
    "docs/index.html", "docs/results.json", "docs/fallback_analysis.json",
    "artifacts/results.json", "artifacts/original_results.json",
    "artifacts/heldout_predictions.csv.gz", "artifacts/original_heldout_predictions.csv.gz",
    "artifacts/models/README.md", "artifacts/models/rf_seed11.joblib",
    "artifacts/models/rf_raw_seed11.joblib",
    "artifacts/models/watch_rf_fallback_seed11.joblib",
    ".github/workflows/ci.yml", ".gitignore", ".gitattributes",
    "scripts/package_release.py",
]


def main():
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("Missing required project files: " + ", ".join(missing))
    results = json.loads((ROOT / "artifacts/results.json").read_text())
    if [r["seed"] for r in results["runs"]] != [11, 23, 37]:
        raise SystemExit("Full three-seed experiment has not completed")
    if any(not all(k in r["models"] for k in
                   ("dummy", "linear", "rf", "rf_raw", "cnn", "transformer", "watch_rf_fallback"))
           for r in results["runs"]):
        raise SystemExit("The full set of measured models is missing")
    target = ROOT.parent / "SenseShift_GitHub_Research.zip"
    root_name = "SenseShift"
    extras = ["data/raw/.gitkeep", "data/processed/.gitkeep"]
    for model in ("cnn", "transformer"):
        for seed in (11, 23, 37):
            extras.append(f"artifacts/models/{model}_seed{seed}.pt")
    missing_extras = [name for name in extras if not (ROOT / name).is_file()]
    if missing_extras:
        raise SystemExit("Missing training checkpoints: " + ", ".join(missing_extras))
    with ZipFile(target, "w", ZIP_DEFLATED, compresslevel=8) as z:
        for name in REQUIRED + extras:
            z.write(ROOT / name, f"{root_name}/{name}")
    with ZipFile(target) as z:
        assert z.testzip() is None
        names = set(z.namelist())
        assert f"{root_name}/src/senseshift/data.py" in names
        assert not any("/raw/data_" in name or "/processed/windows" in name for name in names)
    print(f"{target} ({target.stat().st_size:,} bytes, {len(names)} files)")


if __name__ == "__main__":
    main()

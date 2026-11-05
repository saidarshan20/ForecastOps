"""Keep byte provenance distinct from numerical reproducibility."""
import hashlib
import json
import pandas as pd
from pandas.testing import assert_frame_equal

TIMING_COLUMNS = [f"{kind}_seconds{suffix}" for kind in ["fit", "predict"]
                  for suffix in ["", "_validation", "_test"]]


def check_frozen_provenance(report):
    selection = json.loads((report / "selection.json").read_text(encoding="utf-8"))
    for name, expected in selection["validation_files_sha256"].items():
        if hashlib.sha256((report / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen provenance mismatch: {name}")
    return selection


def compare_csv(expected, reproduced):
    """Same rows/schema; only measured runtime is excluded, numeric tolerance explicit."""
    left, right = pd.read_csv(expected), pd.read_csv(reproduced)
    left = left.drop(columns=TIMING_COLUMNS, errors="ignore")
    right = right.drop(columns=TIMING_COLUMNS, errors="ignore")
    assert_frame_equal(left, right, check_exact=False, rtol=1e-10, atol=1e-7)


def require_new_report_directory(report):
    if report.exists() and any(report.iterdir()):
        raise FileExistsError("Use a new empty output directory; frozen reports must not be overwritten")
    report.mkdir(parents=True, exist_ok=True)

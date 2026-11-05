"""Reproduce frozen evidence into a new directory without replacing the release."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from forecastops.config import ROOT
from forecastops.data.loader import RAW, load_raw, load_processed
from forecastops.evaluation.evidence import check_frozen_provenance, compare_csv
from forecastops.monitoring.replay import run
from scripts.run_validation import main as validation
from scripts.select_and_test import main as test
from scripts.build_serving_artifact import main as build


def snapshot():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for name in ["reports", "artifacts", "runtime", "data/processed"]
            for p in (ROOT / name).rglob("*") if p.is_file()}


def main(output):
    output = output.resolve()
    protected = [ROOT / name for name in ["reports", "artifacts", "runtime", "data", "datasets"]]
    if any(output == p or p in output.parents or output in p.parents for p in protected):
        raise ValueError("Output must be separate from protected evidence")
    output.mkdir(parents=True, exist_ok=False)
    before = snapshot()
    frozen = ROOT / "reports"
    try:
        selection = check_frozen_provenance(frozen)
        manifest = json.loads((frozen / "data_manifest.json").read_text())
        assert hashlib.sha256(RAW.read_bytes()).hexdigest() == manifest["sha256"]
        raw = load_raw().set_index("dteday")["cnt"].astype(float)
        np.testing.assert_array_equal(raw.to_numpy(), load_processed().to_numpy())
        assert raw.index.equals(load_processed().index)
        report = output / "reports"
        validation(report)
        # Verifies frozen hashes, then reproduced validation only, before loading test targets.
        test(report, frozen)
        for name in ["test_predictions.csv", "test_fold_metrics.csv", "test_failures.csv", "test_metrics.csv", "horizon_errors.csv", "model_comparison.csv"]:
            compare_csv(frozen / name, report / name)
        build(report, output / "release")
        old_manifest = json.loads((ROOT / "artifacts/release/manifest.json").read_text())
        new_manifest = json.loads((output / "release/manifest.json").read_text())
        assert old_manifest["selection_sha256"] == hashlib.sha256((frozen / "selection.json").read_bytes()).hexdigest()
        assert new_manifest["selection_sha256"] == hashlib.sha256((report / "selection.json").read_bytes()).hexdigest()
        for field in old_manifest:
            if field not in ["created_at", "selection_sha256"]:
                assert old_manifest[field] == new_manifest[field], f"Release metadata mismatch: {field}"
        original = joblib.load(ROOT / "artifacts/release/model.joblib")
        restored = joblib.load(output / "release/model.joblib")
        np.testing.assert_allclose(original["model"].predict(14, original["history"].index[-1]),
                                   restored["model"].predict(14, restored["history"].index[-1]), rtol=1e-10, atol=1e-7)
        replay = run(output / "monitoring.sqlite", summary_path=output / "monitoring_summary.json")
        saved = json.loads((frozen / "monitoring_summary.json").read_text())
        assert replay["source_checksum"] == saved["source_checksum"]
        for value in [replay, saved]:
            value.pop("generated_at")
        # JSON normalization retains dates, flags, counts, and checksums; only timestamp varies.
        pd.testing.assert_frame_equal(pd.json_normalize({k:v for k,v in saved.items() if k != 'records'}),
                                      pd.json_normalize({k:v for k,v in replay.items() if k != 'records'}),
                                      check_exact=False, rtol=1e-10, atol=1e-7)
        for left, right in zip(saved["records"], replay["records"]):
            for key in left:
                if isinstance(left[key], float):
                    np.testing.assert_allclose(left[key], right[key], rtol=1e-10, atol=1e-7)
                else:
                    assert left[key] == right[key]
        result = {"status": "passed", "selected_model": selection["selected_model"],
                  "python": platform.python_version(), "provenance": "frozen validation byte hashes verified before test reproduction",
                  "excluded_comparison_fields": ["fit_seconds", "predict_seconds", "generated_at", "created_at"],
                  "numeric_tolerance": {"rtol": 1e-10, "atol": 1e-7},
                  "dependencies": {name: metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "statsmodels"]}}
        result["selection_provenance"] = {"frozen_sha256": old_manifest["selection_sha256"],
                                          "reproduced_sha256": new_manifest["selection_sha256"],
                                          "explanation": "Each manifest independently binds its own selection bytes, including its own timestamp and measured validation timings"}
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        after = snapshot()
        (output / "preservation.json").write_text(json.dumps({"unchanged": before == after, "before": before, "after": after}, indent=2) + "\n")
        assert before == after, "Protected files changed during reproduction"
    print(f"Verified frozen experiment; isolated evidence: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; must not exist")
    main(parser.parse_args().output_dir)

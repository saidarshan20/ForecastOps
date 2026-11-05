import json
import hashlib
import pandas as pd
import pytest
from forecastops.evaluation.evidence import compare_csv, check_frozen_provenance


def test_timing_can_vary_but_predictions_cannot(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    pd.DataFrame({"prediction": [123.0], "fit_seconds": [1.0], "fit_seconds_validation": [1.0]}).to_csv(a, index=False)
    pd.DataFrame({"prediction": [123.0], "fit_seconds": [99.0], "fit_seconds_validation": [99.0]}).to_csv(b, index=False)
    compare_csv(a, b)
    pd.DataFrame({"prediction": [124.0], "fit_seconds": [1.0], "fit_seconds_validation": [1.0]}).to_csv(b, index=False)
    with pytest.raises(AssertionError):
        compare_csv(a, b)


def test_frozen_byte_provenance_still_detects_tampering(tmp_path):
    path = tmp_path / "validation.csv"
    path.write_text("mae,fit_seconds\n1,2\n")
    (tmp_path / "selection.json").write_text(json.dumps({"validation_files_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}}))
    check_frozen_provenance(tmp_path)
    path.write_text("mae,fit_seconds\n1,3\n")
    with pytest.raises(ValueError, match="provenance"):
        check_frozen_provenance(tmp_path)

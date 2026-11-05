import hashlib
import importlib.metadata as metadata
import json
from datetime import datetime, timezone
import joblib
import numpy as np
from forecastops.config import ROOT, ARTIFACT_DIR
from forecastops.data.loader import load_processed
from forecastops.features.build import FEATURES
from forecastops.models.registry import make_model


def main(report=None, artifact_dir=None):
    report = report or ROOT / "reports"
    artifact_dir = artifact_dir or ARTIFACT_DIR
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError("Refusing to replace a release; choose a fresh artifact directory")
    selection_path = report / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    series = load_processed().loc[:"2012-10-22"]
    model = make_model(selection["selected_model"]).fit(series)
    expected = np.asarray(model.predict(14, series.index[-1]), dtype=float)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "history": series}, artifact_dir / "model.joblib")
    restored = joblib.load(artifact_dir / "model.joblib")
    np.testing.assert_allclose(expected, restored["model"].predict(14, series.index[-1]), rtol=0, atol=1e-9)
    manifest = {
        "model_id": selection["selected_model"], "version": "v1-20121022", "parameters": selection["parameters"],
        "trained_through": "2012-10-22", "data_start": "2011-01-01", "training_rows": len(series),
        "horizon_limit": 14, "feature_schema": FEATURES if selection["selected_model"] in ["ridge", "hist_gradient_boosting"] else [],
        "data_sha256": json.loads((ROOT / "reports/data_manifest.json").read_text(encoding="utf-8"))["sha256"],
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "dependency_versions": {x: metadata.version(x) for x in ["numpy", "pandas", "scikit-learn", "statsmodels", "joblib"]},
        "created_at": datetime.now(timezone.utc).isoformat(), "forecast_calendar": "consecutive days following historical cutoff",
        "clipping_policy": "clip negative forecasts to zero; preserve floating point", "mode": "historical_demo",
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (report / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Built {manifest['model_id']} {manifest['version']} through {manifest['trained_through']}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.report_dir, args.artifact_dir)

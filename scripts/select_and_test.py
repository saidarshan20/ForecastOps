import hashlib
import importlib.metadata as metadata
import json
from datetime import datetime, timezone
import pandas as pd
from forecastops.config import ROOT
from forecastops.data.loader import load_processed
from forecastops.evaluation.backtest import backtest, summarize, horizon_summary
from forecastops.evaluation.selection import choose
from forecastops.models.registry import NAMES

STARTS = pd.date_range("2012-10-23", periods=5, freq="14D")


def main():
    report = ROOT / "reports"
    metrics_path = report / "validation_metrics.csv"
    if not metrics_path.exists(): raise FileNotFoundError("Run validation first")
    validation = pd.read_csv(metrics_path)
    decision = choose(validation)
    decision.update({"created_at": datetime.now(timezone.utc).isoformat(),
                     "validation_files_sha256": {name: hashlib.sha256((report / name).read_bytes()).hexdigest()
                       for name in ["validation_metrics.csv", "validation_predictions.csv", "validation_fold_metrics.csv", "validation_failures.csv"]},
                     "dependency_versions": {name: metadata.version(name) if name != "statsmodels" or _installed(name) else "unavailable"
                       for name in ["pandas", "numpy", "scikit-learn", "statsmodels"]}})
    selection_path = report / "selection.json"
    if selection_path.exists():
        previous = json.loads(selection_path.read_text(encoding="utf-8"))
        for field in ["selected_model", "parameters", "validation_files_sha256"]:
            if previous[field] != decision[field]: raise ValueError("Frozen selection disagrees with validation; refusing test")
    else:
        selection_path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    # Selection is durable before loading test-period data.
    series = load_processed()
    runs = [backtest(name, series, STARTS) for name in NAMES]
    predictions = pd.concat([x[0] for x in runs], ignore_index=True)
    folds = pd.concat([x[1] for x in runs], ignore_index=True)
    failures = pd.concat([x[2] for x in runs], ignore_index=True) if any(not x[2].empty for x in runs) else pd.DataFrame(columns=["model", "fold", "issue_date", "status", "reason"])
    predictions.to_csv(report / "test_predictions.csv", index=False)
    folds.to_csv(report / "test_fold_metrics.csv", index=False)
    failures.to_csv(report / "test_failures.csv", index=False)
    test = summarize(predictions, folds, NAMES)
    test.to_csv(report / "test_metrics.csv", index=False)
    val_horizons = pd.read_csv(report / "horizon_errors.csv")
    pd.concat([val_horizons[val_horizons["split"] == "validation"], horizon_summary(predictions, "test")]).to_csv(report / "horizon_errors.csv", index=False)
    comparison = validation.merge(test, on="model", suffixes=("_validation", "_test"))
    baseline_test = float(test.set_index("model").loc["seasonal_naive", "mae"])
    comparison["test_improvement_vs_seasonal_pct"] = (baseline_test - comparison.mae_test) / baseline_test * 100 if baseline_test else None
    comparison["selected"] = comparison.model == decision["selected_model"]
    comparison.to_csv(report / "model_comparison.csv", index=False)
    print(comparison[["model", "mae_validation", "mae_test", "selected"]].to_string(index=False))


def _installed(name):
    try: metadata.version(name); return True
    except metadata.PackageNotFoundError: return False


if __name__ == "__main__": main()

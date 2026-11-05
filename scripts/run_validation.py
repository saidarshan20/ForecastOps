from forecastops.config import ROOT
from forecastops.data.loader import load_processed
from forecastops.evaluation.backtest import backtest, summarize, horizon_summary
from forecastops.models.registry import NAMES
from forecastops.evaluation.evidence import require_new_report_directory
import pandas as pd

STARTS = pd.date_range("2012-08-14", periods=5, freq="14D")


def main(report=None):
    series = load_processed()
    report = report or ROOT / "reports"
    require_new_report_directory(report)
    runs = [backtest(name, series.loc[series.index < "2012-10-23"], STARTS) for name in NAMES]
    predictions = pd.concat([x[0] for x in runs], ignore_index=True)
    folds = pd.concat([x[1] for x in runs], ignore_index=True)
    failures = pd.concat([x[2] for x in runs], ignore_index=True) if any(not x[2].empty for x in runs) else pd.DataFrame(columns=["model", "fold", "issue_date", "status", "reason"])
    predictions.to_csv(report / "validation_predictions.csv", index=False)
    folds.to_csv(report / "validation_fold_metrics.csv", index=False)
    failures.to_csv(report / "validation_failures.csv", index=False)
    summarize(predictions, folds, NAMES).to_csv(report / "validation_metrics.csv", index=False)
    horizon_summary(predictions, "validation").to_csv(report / "horizon_errors.csv", index=False)
    print(summarize(predictions, folds, NAMES).to_string(index=False))


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Generate validation in a new, isolated directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args().output_dir)

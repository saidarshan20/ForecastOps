import numpy as np
import pandas as pd


def summarize_replay(frame: pd.DataFrame, selected: str):
    pivot = frame.pivot(index="target_date", columns="model_version", values="absolute_error").sort_index()
    chosen = "selected-v1"
    baseline = "baseline-v1"
    if len(pivot) != 70 or pivot[[chosen, baseline]].isna().any().any():
        raise ValueError("Expected 70 paired day-ahead observations")
    model_errors, baseline_errors = pivot[chosen], pivot[baseline]
    model_roll = model_errors.rolling(28, min_periods=28).mean()
    base_roll = baseline_errors.rolling(28, min_periods=28).mean()
    initial = float(model_roll.iloc[27])
    rows = []
    for i, (date, row) in enumerate(pivot.iterrows()):
        current = float(model_roll.iloc[i]) if i >= 27 else None
        base = float(base_roll.iloc[i]) if i >= 27 else None
        relative = bool(current > 1.2 * base) if current is not None and base > 0 and selected != "seasonal_naive" else False
        # Require a separate full recent window after the initial 28-day window.
        drift = bool(current > 1.2 * initial) if i >= 55 and initial > 0 else False
        rows.append({"target_date": date, "selected_absolute_error": float(row[chosen]),
                     "baseline_absolute_error": float(row[baseline]), "selected_rolling_28_mae": current,
                     "baseline_rolling_28_mae": base, "relative_deterioration": relative,
                     "absolute_drift": drift})
    return {"mode": "historical_replay", "selected_model": selected, "baseline_selected": selected == "seasonal_naive",
            "start": str(pivot.index[0]), "end": str(pivot.index[-1]), "paired_days": len(pivot),
            "selected_day_ahead_mae": float(model_errors.mean()), "baseline_day_ahead_mae": float(baseline_errors.mean()),
            "latest_selected_rolling_28_mae": float(model_roll.iloc[-1]), "latest_baseline_rolling_28_mae": float(base_roll.iloc[-1]),
            "relative_deterioration": rows[-1]["relative_deterioration"], "absolute_drift": rows[-1]["absolute_drift"],
            "zero_reference_policy": "no relative flag when reference MAE is zero",
            "records": rows}

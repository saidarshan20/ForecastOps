from forecastops.models.registry import NAMES, PARAMETERS


def choose(metrics):
    indexed = metrics.set_index("model")
    baseline = indexed.loc["seasonal_naive"]
    if baseline.status != "success":
        raise ValueError("Seasonal baseline must complete validation")
    baseline_mae = float(baseline.mae)
    eligible = [(float(indexed.loc[n].mae), NAMES.index(n), n) for n in NAMES if indexed.loc[n].status == "success"]
    best_mae, _, best = min(eligible)
    improvement = (baseline_mae - best_mae) / baseline_mae if baseline_mae > 0 else 0.0
    selected = best if baseline_mae > 0 and best != "seasonal_naive" and improvement >= 0.01 else "seasonal_naive"
    return {"selected_model": selected, "parameters": PARAMETERS[selected], "baseline_validation_mae": baseline_mae,
            "best_candidate": best, "best_candidate_validation_mae": best_mae,
            "relative_improvement": improvement, "threshold": 0.01,
            "reason": "best eligible candidate passed 1% validation MAE gate" if selected != "seasonal_naive" else "seasonal baseline retained by 1% validation MAE gate"}

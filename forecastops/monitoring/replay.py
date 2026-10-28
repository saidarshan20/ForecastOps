import hashlib
import json
import shutil
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from forecastops.config import ROOT, SEED_DB
from forecastops.data.loader import load_processed
from forecastops.models.registry import make_model
from .store import connect
from .metrics import summarize_replay


def run(path=SEED_DB):
    selection = json.loads((ROOT / "reports/selection.json").read_text(encoding="utf-8"))
    selected = selection["selected_model"]
    series = load_processed()
    checksum = hashlib.sha256((ROOT / "reports/data_manifest.json").read_bytes()).hexdigest()
    now = datetime.now(timezone.utc).isoformat()
    connection = connect(path)
    run_id = f"historical-v1-{checksum[:12]}-{selected}"
    connection.execute("INSERT OR IGNORE INTO monitoring_runs VALUES (?,?,?,?,?)", (run_id, now, None, checksum, "running"))
    for target_date in pd.date_range("2012-10-23", "2012-12-31"):
        issue_date = target_date - pd.Timedelta(days=1)
        past = series.loc[:issue_date]
        issued = []
        for version, name in [("selected-v1", selected), ("baseline-v1", "seasonal_naive")]:
            forecast = float(make_model(name).fit(past).predict(1, issue_date)[0])
            if not np.isfinite(forecast) or forecast < 0: raise ValueError("Invalid replay forecast")
            issued.append((issue_date.date().isoformat(), target_date.date().isoformat(), 1, version, name, forecast, now))
        # Persist both predictions before reading this target's truth.
        connection.executemany("INSERT OR IGNORE INTO forecast_issues VALUES (?,?,?,?,?,?,?)", issued)
        connection.commit()
        actual = float(series.loc[target_date])
        connection.execute("INSERT OR IGNORE INTO actuals VALUES (?,?,?)", (target_date.date().isoformat(), actual, now))
        connection.commit()
    frame = pd.read_sql_query("SELECT f.target_date, f.model_version, ABS(f.prediction-a.actual) AS absolute_error "
                              "FROM forecast_issues f JOIN actuals a USING(target_date) WHERE f.horizon=1", connection)
    summary = summarize_replay(frame, selected)
    summary["generated_at"] = now
    summary["source_checksum"] = checksum
    connection.execute("UPDATE monitoring_runs SET ended_at=?, status='complete' WHERE run_id=?", (now, run_id))
    connection.commit()
    connection.close()
    out = ROOT / "reports/monitoring_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary

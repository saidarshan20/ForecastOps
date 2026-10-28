import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecast_issues (
 issue_date TEXT NOT NULL, target_date TEXT NOT NULL, horizon INTEGER NOT NULL,
 model_version TEXT NOT NULL, model_name TEXT NOT NULL, prediction REAL NOT NULL,
 created_at TEXT NOT NULL,
 UNIQUE(issue_date, target_date, horizon, model_version)
);
CREATE TABLE IF NOT EXISTS actuals (
 target_date TEXT PRIMARY KEY, actual REAL NOT NULL, revealed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS monitoring_runs (
 run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT,
 source_checksum TEXT NOT NULL, status TEXT NOT NULL
);
"""


def connect(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    return connection

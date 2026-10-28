from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.getenv("FORECASTOPS_ARTIFACT_DIR", str(ROOT / "artifacts/release")))
DB_PATH = Path(os.getenv("FORECASTOPS_DB_PATH", str(ROOT / "runtime/monitoring.sqlite")))
SEED_DB = Path(os.getenv("FORECASTOPS_SEED_DB", str(ROOT / "artifacts/monitoring_seed.sqlite")))
RANDOM_SEED = int(os.getenv("FORECASTOPS_RANDOM_SEED", "42"))

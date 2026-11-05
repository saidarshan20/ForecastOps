import hashlib
import json
from datetime import datetime, timezone
from forecastops.config import ROOT
from forecastops.data.loader import RAW, PROCESSED, load_raw


def main():
    if PROCESSED.exists() or (ROOT / "reports/data_manifest.json").exists():
        raise FileExistsError("Prepared data already exists; use scripts.verify_experiment for safe reproduction")
    frame = load_raw()
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PROCESSED, index=False, date_format="%Y-%m-%d")
    manifest = {
        "source": "UCI Bike Sharing Dataset, Hadi Fanaee-T, DOI 10.24432/C5W894, CC BY 4.0",
        "source_url": "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
        "source_file": str(RAW.relative_to(ROOT)).replace("\\", "/"),
        "sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
        "rows": len(frame), "start": str(frame.dteday.iloc[0].date()), "end": str(frame.dteday.iloc[-1].date()),
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    path = ROOT / "reports/data_manifest.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(frame)} days; source SHA-256 {manifest['sha256']}")


if __name__ == "__main__":
    main()

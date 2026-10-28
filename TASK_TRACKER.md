# ForecastOps task tracker

Updated: 2026-10-28 (Asia/Calcutta)

## Resume instructions

Read this file, `ForecastOps_Implementation_Plan.md`, and `docs/experiment_contract.md` first. Continue from the first unchecked item. Keep this tracker current after each phase. Never use held-out test results to change model selection or parameters.

## Progress

- [x] Inspect plan and available data. UCI `datasets/bike+sharing+dataset/day.csv` exists. No application code or Git repository existed at start.
- [x] Phase 0: Freeze experiment contract and decisions.
- [x] Phase 1: Set up package, pinned dependencies, configuration, and isolated Python 3.13 environment.
- [x] Phase 2: Prepare and validate UCI daily data and provenance.
- [x] Phase 3: Build leakage-safe features and recursion.
- [x] Phase 4: Implement five fixed forecasting candidates.
- [x] Phase 5: Run validation backtests and reports.
- [x] Phase 6: Freeze validation-only selection; then run test backtests.
- [x] Phase 7: Build and verify historical serving artifact.
- [x] Phase 8: Run independent 70-day monitoring replay and SQLite seed.
- [x] Phase 9: Build and test FastAPI endpoints, including SQLite-backed monitoring.
- [x] Phase 10: Build dashboard and visually verify in browser.
- [ ] Phase 11: Tests, packaging, and smoke checks. Python suite (20 passed), local HTTP smoke, and browser checks passed; Docker smoke pending because Docker is unavailable.
- [ ] Phase 12: Acceptance review, README, walkthrough, screenshots done; GitHub/Render deployment and deployed smoke pending because no remote/service is available.

## Environment findings

- Python 3.13.5 is installed; Python 3.12 is not listed by `py -0p`. Dockerfile uses 3.13 to match the pinned release.
- Docker command is unavailable.
- Existing UCI `day.csv` SHA-256: `a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be`.
- At start there was no `.git` directory or GitHub remote. GitHub connector search found no installed ForecastOps repository. Render service access is unknown.

## Measured result and checks

- Validation: all five candidates completed 70 forecasts each. Selected Holt-Winters at MAE 793.46 versus seasonal naive 962.06 (17.52% improvement); `reports/selection.json` was written before test data was loaded.
- Held-out test: Holt-Winters MAE 1349.46 versus seasonal naive 1497.66 (9.90% improvement). Ridge test MAE 1255.47 was lower, but selection was not revised.
- Artifact: `artifacts/release/model.joblib` and manifest, fitted through 2012-10-22. Replay: 70 paired one-day targets in `artifacts/monitoring_seed.sqlite`.
- `.venv` installed from `requirements.txt`; `python -m pytest -q` in the pinned venv: **20 passed**. Local `/health`, `/predict` (14 dates), and `/monitoring` passed. Browser dashboard had zero console errors after favicon fix; screenshots in `output/playwright/`.
- Initial local Git commit: `53fa965` (`Implement ForecastOps historical forecasting demo`). No remote is configured. GitHub CLI and Docker are absent; connected GitHub search found no ForecastOps repository.

## Current next step

Docker runtime and deployed Render checks require Docker plus a GitHub remote/Render service. Do not change frozen selection based on test results. If resuming, inspect Git status and `TASK_TRACKER.md`, then continue external deployment with the necessary access. After deployment, add the live URL and deployed smoke results to README/tracker.

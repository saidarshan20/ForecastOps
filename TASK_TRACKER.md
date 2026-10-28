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
- [ ] Phase 11: Tests, packaging, and smoke checks. Python suite (20 passed), local HTTP smoke, and browser checks passed; Docker smoke deferred at the user's request and Docker is unavailable on this machine.
- [ ] Phase 12: Acceptance review, README, expanded beginner walkthrough, screenshots, deployment instructions, and GitHub publication done. Render service creation and deployed smoke are pending.

## Environment findings

- Python 3.13.5 is installed; Python 3.12 is not listed by `py -0p`. Dockerfile uses 3.13 to match the pinned release.
- Docker command is unavailable.
- Existing UCI `day.csv` SHA-256: `a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be`.
- GitHub remote supplied by user: `https://github.com/saidarshan20/ForecastOps.git`, configured as `origin`. It was empty when checked with `git ls-remote`. GitHub connector did not list it among installed repositories, so use local Git for push. Render service access is unknown.

## Measured result and checks

- Validation: all five candidates completed 70 forecasts each. Selected Holt-Winters at MAE 793.46 versus seasonal naive 962.06 (17.52% improvement); `reports/selection.json` was written before test data was loaded.
- Held-out test: Holt-Winters MAE 1349.46 versus seasonal naive 1497.66 (9.90% improvement). Ridge test MAE 1255.47 was lower, but selection was not revised.
- Artifact: `artifacts/release/model.joblib` and manifest, fitted through 2012-10-22. Replay: 70 paired one-day targets in `artifacts/monitoring_seed.sqlite`.
- `.venv` installed from `requirements.txt`; `python -m pytest -q` in the pinned venv: **20 passed**. Local `/health`, `/predict` (14 dates), and `/monitoring` passed. Browser dashboard had zero console errors after favicon fix; screenshots in `output/playwright/`.
- Initial local Git commit: `53fa965` (`Implement ForecastOps historical forecasting demo`), checkpoint commit `a5982c9`. GitHub CLI and Docker are absent.
- Added a beginner-to-advanced request trace in `WALKTHROUGH.md` and current hosting instructions in `docs/DEPLOYMENT.md`. Recommended host: Render Free Docker web service; Render can build the image remotely. Railway and Hugging Face Spaces are documented with current cost caveats.
- Pushed `main` to `https://github.com/saidarshan20/ForecastOps.git`. Remote branch verified at `e5cc2154a62bfb21ae45cc9c90b02b827e9e612c`; local working tree was clean after the push.

## Current next step

Docker runtime checks are deferred at the user's request. Render deployment needs an account/service; follow `docs/DEPLOYMENT.md`, then add the live URL and deployed smoke results to README/tracker. Do not change frozen selection based on test results. If resuming, inspect Git status and this tracker first.

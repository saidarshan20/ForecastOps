# ForecastOps — Complete Implementation Plan

> **Project:** ML forecasting API with trustworthy backtesting, baseline comparisons, containerized serving, and historical error monitoring  
> **Document type:** Step-by-step build guide (planning and commands; not finished application code)  
> **Recommended location in your repository:** `ForecastOps/ForecastOps.md`  
> **Status:** Ready to implement after the decisions below are accepted  
> **Last reviewed:** 27 October 2026

---

## 1. What we're building

**ForecastOps** forecasts daily bike-rental demand for the next **1–14 days** using the **UCI Bike Sharing** daily dataset. An operations user can compare simple baselines with several forecasting models, inspect genuinely out-of-sample results, request a prediction through FastAPI, and view a dashboard showing forecasts and historical error. We choose the model using **validation-only, expanding-window, time-aware backtesting**, then evaluate it once on an **untouched test period**. A separate historical monitoring replay shows whether day-ahead forecasting error deteriorates relative to a seasonal-naive baseline. If no advanced model meaningfully beats the baseline, **the baseline wins**.

### Definition of success

1. A reproducible table compares **five model candidates** (including two baselines) on validation and test metrics.
2. All evaluation windows are chronological; **no future targets or observed future weather** enter prediction inputs.
3. Model selection is **frozen before the held-out test period is evaluated**.
4. `POST /predict` provides 1–14 dated forecasts and explicit **historical model cutoff**.
5. `GET /monitoring` and a dashboard show an **independent 70-day, one-step-ahead historical replay**.
6. The same trained, versioned model artifact works locally and in Docker; a portfolio demo can be deployed on Render Free.
7. All critical behavior is covered by tests, with honest conclusions even when the baseline is best.

### Project architecture

```text
UCI day.csv (2011–2012)
       |
       v
Validate dates & counts -----> Document source / checksum
       |
       v
Chronological train / validation / test
       |
       +--> Five models --> 14-day expanding-window validation
       |                        |
       |                        v
       |                 Freeze winning model class
       |                        |
       |                        v
       |               14-day untouched test backtest
       |                        |
       |                        +--> Metrics & comparison report
       |                        |
       |                        +--> Historical daily monitoring replay
       |                                      |
       |                                      v
       |                                SQLite snapshot
       v
Fit selected model through 2012-10-22
       |
       v
Versioned artifact + historical target series
       |
       v
FastAPI ---> JSON prediction + HTML dashboard
       |
       v
Docker image ---> Render Free (historical demo)
```

**Important distinction:** Fourteen-day backtesting, one-day-ahead monitoring, and a fixed historical API artifact serve different purposes. Do not mix their results or imply this is a live 2026 demand-forecasting service.

---

## 2. What to expect

### Scope locked for version 1

| Area | Decision | Reason |
|---|---|---|
| Forecast target | Daily total bike rentals (`cnt`) | Clear, numeric operational demand signal |
| Data | UCI Bike Sharing `day.csv`; use only `dteday` and `cnt` | Simple time-series demo without accidental future covariates |
| Time period | 2011-01-01 to 2012-12-31 (731 days) | Complete historical daily series |
| Horizon | Integer 1–14 days | Useful multistep prediction without excessive scope |
| Models | Seasonal naive, last-value naive, Holt-Winters, Ridge, histogram gradient boosting | Progression from simple to more complex |
| Model selection | Validation MAE, with a **1% relative improvement gate** over seasonal naive | Rejects negligible gains from complexity |
| Main evaluation | Five non-overlapping 14-day validation windows, then five test windows | Same horizon and honest chronological backtest |
| Monitoring | Separate 70-day one-step-ahead replay | Gives sufficient observations for a 28-day rolling MAE |
| API | FastAPI + Pydantic | Typed prediction contract and automatic API docs |
| Charts | Small HTML/CSS/JavaScript page served by FastAPI | Keeps attention on ML/backend quality |
| Storage | SQLite for historical monitoring records | Zero external database dependency |
| Deployment | Docker on Render Free | Easy public portfolio demonstration |
| Retraining | Only controlled **offline** refits during evaluation/replay | Avoids creating a production automation system |

### Technology

- **Python 3.12** — implementation runtime.
- **pandas, NumPy** — validation, chronological features, metrics, reports.
- **scikit-learn** — Ridge, `HistGradientBoostingRegressor`, preprocessing.
- **statsmodels** — Holt-Winters exponential smoothing.
- **FastAPI, Pydantic, Uvicorn** — prediction service.
- **SQLite** (stdlib `sqlite3`) — local replay/error history.
- **pytest, httpx** — unit and API integration tests.
- **Docker** — reproducible hosting.
- **pip-tools** — a resolved, pinned dependency set used for both training and serving.

**Realistic time budget:** approximately **35–55 focused hours** if this is your first complete forecasting-service project. A useful breakdown is setup/data (5–7 h), features/models (9–14 h), backtesting/selection (8–12 h), monitoring/API/dashboard (8–14 h), and tests/deployment/docs (5–8 h). These are planning estimates, not deadlines.

**Not in version 1:** LLMs or chat; GPUs; real-time rental-data integration; authentication; external databases; automatic scheduled retraining; push/email alerts; complex frontend frameworks; multiple datasets; production SLAs.

### Expected end products

- `reports/model_comparison.csv`, `reports/validation_metrics.csv`, `reports/test_metrics.csv`
- `reports/backtest_predictions.csv`, `reports/horizon_errors.csv`, `reports/monitoring_summary.json`
- One selected, versioned model artifact plus metadata and bounded historical input data
- `GET /health`, `GET /model`, `POST /predict`, `GET /monitoring`, `/docs`, and a dashboard at `/`
- SQLite historical monitoring snapshot and reproducible replay command
- `Dockerfile`, tests, README, screenshots, live-demo link, and an honest model-selection conclusion

---

## 3. Build steps — implement in this order

### Phase 0 — Freeze the experiment contract **before coding**

**Action**

Write a short `docs/experiment_contract.md` agreeing to the following fixed rules. This prevents changing the rules after seeing test results.

**Exact partitions** (indices use zero-based positions after sorting):

| Split | Indices (inclusive) | Dates | Count | Intended use |
|---|---|---|---:|---|
| Initial training | 0–590 | 2011-01-01 to 2012-08-13 | 591 | First forecast origin |
| Validation | 591–660 | 2012-08-14 to 2012-10-22 | 70 | Select candidate |
| Test | 661–730 | 2012-10-23 to 2012-12-31 | 70 | Final untouched comparison |

**Non-overlapping 14-day windows:**

| Window | Validation target dates | Test target dates |
|---|---|---|
| 1 | Aug 14–Aug 27, 2012 | Oct 23–Nov 05, 2012 |
| 2 | Aug 28–Sep 10, 2012 | Nov 06–Nov 19, 2012 |
| 3 | Sep 11–Sep 24, 2012 | Nov 20–Dec 03, 2012 |
| 4 | Sep 25–Oct 08, 2012 | Dec 04–Dec 17, 2012 |
| 5 | Oct 09–Oct 22, 2012 | Dec 18–Dec 31, 2012 |

At each 14-day origin: (1) fit using actuals **strictly before** the target window, (2) issue all 14 predictions without peeking at actuals inside that window, (3) save predictions, (4) only then reveal the window's actual values before the next origin. Within a window, regression features use **previous predictions** when actuals are not yet available.

**Selection rule:** `relative_improvement = (baseline_validation_MAE - candidate_validation_MAE) / baseline_validation_MAE`. Select the candidate with the lowest validation MAE **only if** it beats the seasonal-naive validation MAE by at least **1%**; otherwise select seasonal naive. Fix the parameters and selection **before** test evaluation. Resolve exact validation-MAE ties deterministically using this predefined preference order: seasonal naive → last-value naive → Holt-Winters → Ridge → histogram gradient boosting (simpler model first). If baseline MAE is zero, retain the baseline and avoid division by zero. Do not optimize on test performance.

**Files**: `docs/experiment_contract.md`, `docs/decisions.md`.

**Commands**: none (document and commit the rules).

**Done when:** The split dates, target horizon, selection threshold, metric definitions, and replay protocol are unambiguous.

---

### Phase 1 — Set up the repository and dependencies

**Action**

1. Create a Python 3.12 virtual environment using **PowerShell** on Windows.
2. Create a Python package and organize model, evaluation, API, and monitoring logic separately.
3. Put direct dependencies into `requirements.in`; resolve and pin transitives into `requirements.txt` using `pip-compile`.
4. Commit the lockfile. Use **that same resolved environment** for scripts, tests, API, and Docker. Avoid fetching packages at container startup.
5. Set defaults for paths, log level, port, and random seed from environment variables.
6. Add `.gitignore` for `.venv/`, `__pycache__/`, temp files, local databases, and raw untracked downloads. **Do not ignore** intentional release snapshots and model artifacts included for demo deployment.

**Suggested project structure**

```text
ForecastOps/
├── ForecastOps.md                  # This plan
├── README.md
├── .gitignore
├── .env.example
├── requirements.in
├── requirements.txt               # pip-compile lock/pins
├── pyproject.toml                 # Package metadata for editable install
├── Dockerfile
├── .dockerignore
├── data/
│   ├── raw/                       # Original zip / day.csv (untracked)
│   └── processed/                 # Cleaned series (generated)
├── docs/
│   ├── experiment_contract.md
│   ├── decisions.md
│   └── data_attribution.md
├── forecastops/
│   ├── __init__.py
│   ├── config.py
│   ├── data/{__init__.py,validation.py,loader.py}
│   ├── features/{__init__.py,build.py}
│   ├── models/{__init__.py,baselines.py,statistical.py,regression.py,registry.py}
│   ├── evaluation/{__init__.py,backtest.py,metrics.py,selection.py}
│   ├── monitoring/{__init__.py,replay.py,store.py,metrics.py}
│   └── api/{__init__.py,main.py,schemas.py,service.py}
├── scripts/
│   ├── __init__.py
│   ├── prepare_data.py
│   ├── run_validation.py
│   ├── select_and_test.py
│   ├── build_serving_artifact.py
│   └── replay_monitoring.py
├── web/{index.html,app.js,styles.css}
├── artifacts/                     # Selected model + metadata; versioned releases
├── reports/                       # Evaluation and monitoring outputs
├── runtime/                       # Mutable local SQLite; gitignored
└── tests/
    ├── test_data.py
    ├── test_features.py
    ├── test_baselines.py
    ├── test_backtest.py
    ├── test_selection.py
    ├── test_monitoring.py
    └── test_api.py
```

**PowerShell commands** (from the directory that will contain `ForecastOps/`):

```powershell
mkdir ForecastOps
cd ForecastOps
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip pip-tools
# After creating requirements.in and pyproject.toml:
pip-compile --output-file requirements.txt requirements.in
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python --version
```

If PowerShell prevents activation, use `.\.venv\Scripts\python.exe` directly rather than changing company-wide execution policies.

**Configuration defaults**

| Variable | Default | Purpose |
|---|---|---|
| `FORECASTOPS_ARTIFACT_DIR` | `artifacts/release` | Fitted model + manifest |
| `FORECASTOPS_DB_PATH` | `runtime/monitoring.sqlite` | Writable local SQLite file |
| `FORECASTOPS_SEED_DB` | `artifacts/monitoring_seed.sqlite` | Historical snapshot copied only if DB missing |
| `FORECASTOPS_LOG_LEVEL` | `INFO` | Structured/ordinary app logs |
| `FORECASTOPS_RANDOM_SEED` | `42` | Deterministic stochastic learners |
| `PORT` | `8000` locally; `10000` in Docker/Render | API bind port |

**Done when:** A clean environment installs successfully, `python -m pytest` starts, package imports work, and scripts know their configured file locations.

---

### Phase 2 — Download and validate the dataset

**Action**

Download the source archive from UCI. Use **only** `day.csv`; select `dteday` as the date and `cnt` as the target. Keep the original archive information/attribution in the docs.

**PowerShell commands:**

```powershell
New-Item -ItemType Directory -Force data\raw | Out-Null
Invoke-WebRequest -Uri 'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip' -OutFile 'data\raw\bike_sharing.zip'
Expand-Archive -Path 'data\raw\bike_sharing.zip' -DestinationPath 'data\raw\extracted' -Force
# If the UCI download URL changes, use the Download button on the source page.
Get-ChildItem 'data\raw\extracted' -Recurse -Filter 'day.csv'
python -m scripts.prepare_data
```

The preparation script must locate the extracted `day.csv` (do not assume a fixed inner zip folder), parse/sort by date, keep two columns, and fail loudly on bad data.

**Validation rules:**

- Exactly **731 rows** and **731 unique dates**.
- Date range exactly **2011-01-01 through 2012-12-31**, with one consecutive observation per day.
- `cnt` has **no missing values**, is finite, integral, and `>= 0`.
- Reject duplicated/invalid dates and unexpected missing values; **do not silently fill gaps or interpolate**.
- Keep only date and target for features. **Do not use** future-known-only-in-hindsight weather (`temp`, `hum`, etc.) or components (`casual`, `registered`).
- Compute a SHA-256 checksum on the original `day.csv`; save its hash and source in `reports/data_manifest.json`.

**Source and license:** UCI Bike Sharing Dataset, Hadi Fanaee-T, DOI `10.24432/C5W894`, **CC BY 4.0**. Attribute it in README and dashboard. Official page: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

**Files:** `forecastops/data/loader.py`, `forecastops/data/validation.py`, `scripts/prepare_data.py`, `data/processed/daily_rentals.csv`, `reports/data_manifest.json`, `docs/data_attribution.md`, `tests/test_data.py`.

**Done when:** The valid original dataset passes; duplicates, gaps, negative targets, non-finite targets, and wrong row counts each produce clear failures.

---

### Phase 3 — Implement **leakage-safe** feature generation

**Action**

For supervised learners only, create features available **at forecast issue time**:

| Feature | Meaning | Availability rule |
|---|---|---|
| `lag_1` | Rental count 1 day earlier | Known for training; predicted during recursive steps if needed |
| `lag_7` | Rental count 7 days earlier | Same rule |
| `lag_14` | Rental count 14 days earlier | Same rule |
| `lag_28` | Rental count 28 days earlier | Same rule |
| `rolling_mean_7` | Mean of the previous 7 target observations | **Shift by 1 first**; never include today's target |
| `rolling_mean_28` | Mean of previous 28 target observations | **Shift by 1 first** |
| `weekday` | Forecast-date day of week | Calendar-known |
| `month` | Forecast-date month | Calendar-known |

- Initial supervised rows without 28 days of history are excluded **inside the relevant training partition**; no future data should be used to fill them.
- For one prediction date, feature-building receives only an ordered history ending on the prior date plus the requested target date.
- For the next date in a 14-day forecast, append the **previous prediction** (not the unrevealed actual) before recomputing lag/rolling features.
- Use one shared recursive prediction function for **both Ridge and HistGradientBoosting**, including inference via API.
- Use a scikit-learn `Pipeline` to fit Ridge scaling **only on the training window**; never fit preprocessing on validation/test.
- Never compute rolling or lag features from a globally concatenated train+future-actual table during prediction.

**Files:** `forecastops/features/build.py`, `forecastops/models/regression.py`, `tests/test_features.py`.

**Commands:** `python -m pytest tests/test_features.py -q`

**Critical invariance test:** Run a forecast, alter every actual target value *after* the forecast issue date, and rerun. **Every originally issued forecast must stay identical.** Apply this test to all five forecasting strategies where appropriate, including seasonal naive.

**Done when:** Features are chronological, recursive predictions are repeatable, and the future-actual mutation test passes.

---

### Phase 4 — Implement the five fixed forecasting candidates

**Action**

Define a uniform adapter behavior: `fit(history_ending_at_origin)` and `predict(horizon, origin_date) -> exactly horizon ordered daily predictions`. Forecasts must be numeric and finite; clip negative results to **zero before metric computation**, without rounding. Keep floating-point forecasts for fair comparison and API transparency.

| Candidate | Fixed v1 design | Important edge cases |
|---|---|---|
| **Seasonal naive** | Forecast same weekday from 7 days earlier | For steps 8–14, recursively repeat earlier **forecasted** weekdays (not actuals in the target window) |
| **Last-value naive** | Repeat the last known value for all 14 days | Always nonnegative with valid source data |
| **Holt-Winters** | Additive trend, **damped trend**, additive weekly seasonality (`seasonal_periods=7`) | Fit only on history at origin; handle convergence/optimizer errors visibly |
| **Ridge** | `StandardScaler` + `Ridge(alpha=10.0)` (fixed initial choice) | Training-only scaling; lag/rolling recursion |
| **HistGradientBoostingRegressor** | `max_iter=200`, `learning_rate=0.05`, `max_leaf_nodes=15`, `random_state=42` (fixed initial choices) | Fixed seed; lag/rolling recursion |

These hyperparameters are **pre-registered demonstration defaults**, not claims of optimal settings. Do not tune based on test performance. For Holt-Winters, use `statsmodels.tsa.holtwinters.ExponentialSmoothing` with `trend='add'`, `damped_trend=True`, `seasonal='add'`, `seasonal_periods=7`, `initialization_method='estimated'`.

**Failures:** If a candidate fails on a window, record the model, origin, exception summary, and status; do **not** silently assign invented forecast values. A failed model is ineligible for selection unless every scheduled validation window completes successfully. Baselines should always run on valid input.

**Files:** `forecastops/models/baselines.py`, `statistical.py`, `regression.py`, `registry.py`, `tests/test_baselines.py`.

**Commands:** `python -m pytest tests/test_baselines.py -q`

**Done when:** Every model returns 1–14 ordered, finite, nonnegative daily predictions on known historical cutoffs, or reports a testable explicit failure.

---

### Phase 5 — Build **14-day expanding-window validation backtesting**

**Action**

Implement backtesting with a strict API: `backtest(model, series, origins, horizon=14)`.

For each **validation** origin:

1. Collect available targets through **the day before** the next target date.
2. Fit the candidate on those targets (including previously revealed completed windows).
3. Predict 14 days **all at once** with no actuals revealed during that horizon.
4. Persist `model`, `fold`, `issue_date`, `target_date`, `horizon_step`, `prediction` before matching actuals.
5. Only after all 14 predictions exist, join the true 14 observed targets for scoring.
6. Advance to the next non-overlapping window and extend history with the now-revealed actuals.

**Validation outputs**

- `reports/validation_predictions.csv`: one record per `(model, origin, horizon_step)`.
- `reports/validation_metrics.csv`: metrics by model and a separate fold-level table.
- `reports/validation_failures.csv`: candidate errors with reason and window.
- `reports/horizon_errors.csv`: MAE by horizon step, plus days 1–7 and 8–14.

**Metric definitions**

| Metric | Definition / caveat |
|---|---|
| MAE | Mean absolute error; **primary selection metric** |
| RMSE | Square root of mean squared error; emphasizes large misses |
| WAPE | `sum(abs(actual - predicted)) / sum(abs(actual)) × 100`; report undefined when denominator is 0 |
| Baseline improvement | `(baseline MAE - model MAE) / baseline MAE × 100`; smaller/negative means no improvement |
| Model runtime | Fit seconds and prediction seconds, measured per fold and summarized |

Show **per-fold**, aggregate, and **per-horizon** results. Overall metrics should be computed from pooled out-of-sample predictions, not an accidentally unequal average over incomplete folds.

**Files:** `forecastops/evaluation/backtest.py`, `metrics.py`, `scripts/run_validation.py`, `tests/test_backtest.py`.

**Commands:**

```powershell
python -m scripts.run_validation
python -m pytest tests/test_backtest.py -q
```

**Done when:** Each successful model has **70 distinct validation forecasts**, all from 14-day issue-time origins with no prediction seeing its own target or any target later than the issue date.

---

### Phase 6 — Freeze selection, then run the **untouched test**

**Action**

1. Read validation outputs; identify lowest **fully successful** candidate MAE.
2. Apply the fixed **1% relative improvement gate** against seasonal naive.
3. Write immutable `reports/selection.json` with chosen model, fixed parameters, reason, threshold, timestamp, dependency versions, and the exact validation files/checksums used.
4. Freeze selection. **Do not view or use test metrics when deciding the model.**
5. Evaluate the selected model **and all predeclared candidates** on the final 70 days using the same five 14-day origins; training history starts with all train+validation actuals and expands *after each completed test window*.
6. Generate `reports/test_metrics.csv`, `reports/test_predictions.csv`, final `reports/model_comparison.csv`, and comparison figures.
7. Explain whether the selected model actually beat the seasonal baseline **on test**. If not, say so explicitly. Don't revise model/threshold/hyperparameters using that test result.

**Files:** `forecastops/evaluation/selection.py`, `scripts/select_and_test.py`, `reports/selection.json`, `reports/model_comparison.csv`, `tests/test_selection.py`.

**Commands:**

```powershell
python -m scripts.select_and_test
python -m pytest tests/test_selection.py -q
```

**Expected table shape** (numbers are **not** known until you run the experiment):

| Model | Validation MAE | Test MAE | Test RMSE | Test WAPE | Test vs seasonal baseline | Selection status |
|---|---:|---:|---:|---:|---:|---|
| Seasonal naive | TBD | TBD | TBD | TBD | Reference | Selected / not selected |
| Last-value naive | TBD | TBD | TBD | TBD | TBD | Selected / not selected |
| Holt-Winters | TBD | TBD | TBD | TBD | TBD | Selected / not selected |
| Ridge | TBD | TBD | TBD | TBD | TBD | Selected / not selected |
| HistGradientBoosting | TBD | TBD | TBD | TBD | TBD | Selected / not selected |

**Done when:** Selection has a durable validation-only record, test folds are truly untouched until selection, failed models are labeled, and the comparison table contains no invented measurements.

---

### Phase 7 — Prepare a **versioned historical serving artifact**

**Action**

For the main demonstration, **fit the frozen selected model using data from 2011-01-01 through 2012-10-22 only** (the train+validation period). This is distinct from the expanding fits used to score test folds.

Artifact contents:

- Trained model or stored baseline configuration.
- Historical target series up to **2012-10-22**, enough for lag/rolling or baseline forecasts.
- `manifest.json`: `model_id`, `version`, fixed hyperparameters, `trained_through`, `data_start`, horizon limit, feature schema, training-row count, data checksum, dependency versions, validation selection evidence, and creation timestamp.
- Explicit record of forecast calendar behavior and clipping policy.

**Default serving dates:** A request with `horizon=14` returns **2012-10-23 through 2012-11-05**, regardless of today's date. A request with `horizon=1` returns **2012-10-23**.

**Optional version 2, only after all evaluation is finalized:** Produce a **separately labeled** artifact trained on the full 731-day dataset (cutoff **2012-12-31**, projections starting **2013-01-01**). Never replace the primary evaluated artifact silently or use this later-fit artifact to generate earlier test claims.

**Files:** `scripts/build_serving_artifact.py`, `artifacts/release/`, `reports/artifact_manifest.json`.

**Commands:** `python -m scripts.build_serving_artifact`

**Done when:** Restarting Python and loading the persisted artifact yields the same forecasts as the in-memory model; the cutoff/date range and version are explicitly recorded.

---

### Phase 8 — Build **separate daily one-step-ahead error monitoring**

**This fixes the main issue in the original plan:** Five 14-day forecast windows provide **only five** horizon-1 predictions. They are **not** a 70-point day-ahead monitoring series.

**Action: independent daily replay**

For **each of the 70 test dates** (2012-10-23 to 2012-12-31):

1. Set `issue_date = target_date - 1 day`.
2. Make **only a one-day-ahead forecast** for the frozen selected model type and the seasonal-naive baseline.
3. Refit each forecasting strategy on **only the actual history known through the issue date**, using the same fixed hyperparameters; do **not** reselect or retune the model.
4. Save each prediction and its issue/target date **before** revealing that target's actual.
5. Reveal that target's actual; attach it to previously issued predictions; calculate absolute error.
6. Continue to the following day, now allowed to use the newly revealed actual.

**Why refit daily here?** This explicitly simulates a rolling-origin system that receives daily actuals and refits its fixed model configuration at each issue date. It is an **offline operational replay**, **not** automatic retraining of the hosted API's fixed artifact. The dashboard must label it accordingly. Do **not** compare the 70 one-step-ahead errors as if they were the same forecast task as the five full 14-step test windows.

**SQLite suggested tables**

| Table | Main fields | Constraint |
|---|---|---|
| `forecast_issues` | `issue_date`, `target_date`, `horizon`, `model_version`, `model_name`, `prediction`, `created_at` | Unique on `(issue_date,target_date,horizon,model_version)` |
| `actuals` | `target_date`, `actual`, `revealed_at` | Unique `target_date` |
| `monitoring_runs` | `run_id`, `started_at`, `ended_at`, `source_checksum`, `status` | Primary key `run_id` |

*Each baseline/model row needs a distinct version identifier so keys cannot collide.* Use an upsert or explicit uniqueness check. Re-running the replay with the same dataset and model version **must not duplicate rows**.

**Monitoring calculations**

- **Day-ahead MAE** for the selected model and seasonal naive using the **same matched 1-day horizons**.
- **Rolling 28-target-day MAE**, defined only after **28 comparable paired actuals** are available (first valid result: day 28 of the replay).
- Flag `relative_deterioration` when selected rolling MAE is `> 1.20 × baseline rolling MAE`, provided the baseline rolling MAE is greater than zero.
- If the selected model **is** seasonal naive, comparing it with itself cannot indicate deterioration. Instead show `baseline_selected = true` and use an **absolute drift flag**: recent 28-day MAE `> 1.20 ×` its initial 28-day MAE (only when enough history exists to compare full windows), with an explicit zero-reference handling policy.
- For non-baseline selection also show change from the model's initial valid 28-day rolling MAE; don't make it replace the baseline comparison.
- Provide timestamps and counts; don't call a replay-generated flag a live operational alert.

**Snapshot policy**

- Generate `artifacts/monitoring_seed.sqlite` as a **precomputed, read-only source snapshot** after replay.
- On application startup, copy the snapshot to the writable `runtime/monitoring.sqlite` **only when the runtime DB does not exist**.
- The API reads the writable copy; local changes persist locally. On Render Free they are reset if the filesystem is recycled.

**Files:** `forecastops/monitoring/replay.py`, `store.py`, `metrics.py`, `scripts/replay_monitoring.py`, `artifacts/monitoring_seed.sqlite`, `reports/monitoring_summary.json`, `tests/test_monitoring.py`.

**Commands:**

```powershell
python -m scripts.replay_monitoring
python -m pytest tests/test_monitoring.py -q
```

**Done when:** There are **70 paired day-ahead forecasts per model** (selected and baseline), 28-day rolling metrics start only at day 28, no history is revealed early, and replaying twice does not double the database row count.

---

### Phase 9 — Build and validate the FastAPI service

**Action**

Implement and document these endpoints:

| Route | Purpose | Key contract |
|---|---|---|
| `GET /health` | Readiness/health | `200` when model and required artifacts load; `503` with safe reason if not |
| `GET /model` | Describe the deployed artifact | Model name/version, cutoff, horizon limit, validation-selection rule, historical-demo warning |
| `POST /predict` | Forecast 1–14 days from fixed historical cutoff | Validates integer `horizon`, default optional `14`; returns dated, finite, nonnegative forecasts |
| `GET /monitoring` | Read saved **historical replay** | Last 28-day MAE, selected/baseline comparison, flags, full observed date range, `mode='historical_replay'` |
| `GET /` | HTML dashboard | Renders charts from API/report data |
| `GET /docs` | OpenAPI interactive documentation | Automatically served by FastAPI |

**Example request** (illustrative JSON contract):

```json
{"horizon": 14}
```

**Expected response schema** (the model must generate the numerical values; do not hard-code dummy predictions):

| JSON field | Type | Example or contract |
|---|---|---|
| `model_name` | string | Name from frozen selection artifact |
| `model_version` | string | Release version from manifest |
| `mode` | string | `historical_demo` |
| `trained_through` | ISO date string | `2012-10-22` |
| `horizon` | integer | `14` |
| `forecast_start` | ISO date string | `2012-10-23` |
| `forecast_end` | ISO date string | `2012-11-05` |
| `predictions` | array of objects | **14** objects containing `date` (ISO date string) and **numeric** `value` (float) |
| `disclaimer` | string | `Historical demonstration; not a forecast for the current date.` |

**Response rule:** For horizon 14, the output must contain exactly 14 daily values generated by the persisted model. All predictions must be finite, nonnegative numbers, not strings or placeholder text.

**Request validation:** Reject `horizon=0`, `15`, fractions, strings (unless deliberately and consistently parsed by the schema), booleans, NaN/Infinity, and unsupported fields as appropriate. FastAPI/Pydantic should return `422` for invalid requests. Avoid allowing callers to override the training cutoff in v1.

**Serving boundary:** `/predict` uses the **persisted selected artifact**, *not* a hidden refit over 2012-12-31, and not any target actuals from the test period.

**Files:** `forecastops/api/main.py`, `service.py`, `schemas.py`, `tests/test_api.py`.

**PowerShell commands:**

```powershell
uvicorn forecastops.api.main:app --reload --host 127.0.0.1 --port 8000
# In a second PowerShell window:
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/model
Invoke-RestMethod http://127.0.0.1:8000/monitoring
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body '{"horizon":14}'
```

**Done when:** Valid predictions use historical dates and artifact metadata, invalid horizons give `422`, `/docs` works, monitoring returns the saved replay, and missing model files prevent readiness (`503`).

---

### Phase 10 — Create a focused dashboard

**Action**

Create a mobile-friendly single-page HTML dashboard served by FastAPI. Use plain JavaScript and an optional small chart library (or native SVG/Canvas); avoid building a complex SPA.

**Dashboard panels**

1. **Project banner:** `Historical demo · UCI Bike Sharing 2011–2012 · trained through 2012-10-22`.
2. **Forecast:** Request horizon 1–14; line chart of the selected artifact's returned future **historical** dates.
3. **Model scoreboard:** Validation MAE, test MAE/RMSE/WAPE, baseline-relative test improvement, winning-model explanation, failed-model badges.
4. **Per-horizon error:** Compare days 1–7 versus days 8–14; optionally show 1–14 MAE line chart.
5. **Historical monitor:** 70-day replay chart; paired daily absolute errors, rolling 28-day MAEs and flags.
6. **Model metadata and limitations:** selected model/version, source attribution, data checksum link, date cutoff, deployment status, and **not live** disclaimer.

**Accuracy rules:** Label plots `Backtest (14-day horizon)` or `Monitoring replay (1-day horizon)`; never merge them into one unidentified "accuracy" curve. Do not render TBD placeholders as real results.

**Files:** `web/index.html`, `web/app.js`, `web/styles.css`, route in `forecastops/api/main.py`.

**Commands:** Launch Uvicorn and visit `http://127.0.0.1:8000/`.

**Done when:** A new reader can see which model was selected, whether it beats the baseline, forecast dates, test performance, and replay health **without reading code**.

---

### Phase 11 — Test the full pipeline and containerize

**Action**

Write tests that enforce behavior rather than just import success. Include deterministic seeds, a stable artifact load test, API validation, monitoring replay idempotency, and Docker runtime readiness.

**Test matrix**

| Test | Required assertion |
|---|---|
| Data contracts | 731 days, fixed date range, no duplicates/gaps/negative/nonfinite targets |
| Split boundaries | Train/validation/test disjoint and chronological; test 70 days |
| Lag & rolling shift | `rolling_mean_7` on date T uses only T−7 to T−1 |
| Recursion | Every forecast beyond step 1 uses earlier predictions if needed |
| **Future-target mutation** | Changing all actuals *after* issue date leaves issued forecasts unchanged |
| Baselines | Weekly recursion is correct for steps 8–14; last-value repeats last known observation |
| Fold rules | Exactly five 14-day folds; no truth available during issuing phase |
| Metrics | Hand-check MAE, RMSE, WAPE and safe zero denominators |
| Selection | Validation-only, 1% gate, exact 1% edge case, failure handling, deterministic ties |
| Untouched test | Test targets cannot influence selection/parameters |
| Daily replay | 70 distinct day-ahead targets; no early actuals; 28-day MAE starts on day 28 |
| Replay rerun | Unique rows remain unique and values stable |
| Artifact | Reload forecast is deterministic, cutoff/version preserved |
| API | 1 and 14 accepted; 0, 15, invalid types rejected; dates correct |
| Readiness | Missing manifest/model/required snapshot causes `503` |
| Docker | Boot, `/health`, `/predict`, `/monitoring`, `/docs`, dashboard work |

**Run the complete local path**

```powershell
python -m pytest -q
python -m scripts.prepare_data
python -m scripts.run_validation
python -m scripts.select_and_test
python -m scripts.build_serving_artifact
python -m scripts.replay_monitoring
python -m pytest -q
```

Scripts should be explicitly classified:

| Command | Generates/modifies | Only reads |
|---|---|---|
| `prepare_data` | Processed CSV, data manifest | Raw dataset |
| `run_validation` | Validation predictions/metrics | Processed CSV, experiment contract |
| `select_and_test` | Selection, held-out test and comparison files | Validation outputs, processed CSV |
| `build_serving_artifact` | Release model and manifest | Frozen selection, train+validation data |
| `replay_monitoring` | SQLite seed and monitor report | Frozen model configuration, chronological series |
| `uvicorn ...` | At most local runtime SQLite seed copy and transient logs | **Existing** release artifact, reports, seed snapshot |

**Docker requirements**

- Base on a compatible **Python 3.12** image.
- Install **the same pinned `requirements.txt`** used during experiments.
- Install application code, static assets, **prebuilt** model release, reports, and SQLite seed snapshot.
- **Do not train or run backtests on container startup.**
- Set production host to `0.0.0.0` and port to `${PORT:-10000}`.
- Ensure runtime database directory is writable by the app process.
- Fail readiness if essential artifacts cannot be loaded; never report success with an untrained placeholder.
- Build from an explicit release manifest/checksum. Exclude raw source archives and development caches if not necessary for serving.

**PowerShell smoke checks**

```powershell
docker build -t forecastops:local .
docker run --rm -p 10000:10000 -e PORT=10000 forecastops:local
# In another terminal:
Invoke-RestMethod http://127.0.0.1:10000/health
Invoke-RestMethod -Uri http://127.0.0.1:10000/predict -Method Post -ContentType 'application/json' -Body '{"horizon":7}'
Invoke-RestMethod http://127.0.0.1:10000/monitoring
```

**Done when:** A clean checkout (with included/rebuilt release artifacts) reproduces expected outputs; all automated checks pass; the Docker container loads the release without retraining and serves valid predictions.

---

### Phase 12 — Perform final acceptance review

**Action**

Complete this checklist before marking the project finished:

- [ ] Dataset source, CC BY 4.0 attribution, and checksum are documented.
- [ ] Splits are **591 / 70 / 70**, with no overlap and exact dates.
- [ ] Every model is measured against seasonal naive on the **same 14-day folds**.
- [ ] Actuals inside a forecast window are hidden until after the forecast is issued.
- [ ] Future-target mutation tests pass across models.
- [ ] Selection file is based **only on validation**, with the 1% rule.
- [ ] Untouched test evaluation runs after selection; no test-driven retuning.
- [ ] Both overall and horizon-specific validation/test errors are published.
- [ ] If sophisticated models lose to baseline, README says so plainly.
- [ ] Serving artifact cutoff is **2012-10-22**, and predictions use Oct–Nov 2012 dates.
- [ ] Independent monitoring uses **70 daily one-day-ahead** forecasts.
- [ ] Monitor replay is reproducible/idempotent and handles baseline-selected mode.
- [ ] API checks, persistence check, and container smoke tests pass.
- [ ] A deployed `POST /predict` returns a valid response and visibly identifies historical mode.

**Files:** `README.md`, `reports/`, `tests/`, `docs/`, deployment configuration.  
**Commands:** `python -m pytest -q`, Docker smoke checks, and one actual deployed `POST /predict`.  
**Done when:** All items pass; the GitHub repo and deployment tell the same, verifiable story.

---

## 4. Beginner walkthrough — how the completed system works

### Flow A: training, evaluation, and model selection

```text
Daily rentals (date, cnt)
       |
Check dates, missing values, contiguous daily rows
       |
Chronological partitions: 591 train / 70 validation / 70 test
       |
For each model and validation origin:
  Fit on known past --> issue 14-day forecast --> then reveal actuals
       |
Calculate pooled validation MAE / RMSE / WAPE + per-horizon MAE
       |
Apply baseline + 1% improvement gate, freeze model identity
       |
Run identical 14-day test backtest without changing selection
       |
Publish honest model comparison and selection evidence
```

**Training vs prediction:** Training (`fit`) uses past observed rental counts to estimate parameters. Prediction (`predict`) generates target dates without using their unrevealed actual counts. In recursive regression, a predicted day becomes the history value used to forecast later days in the same 14-day window.

### Flow B: what happens on a browser/API request

```text
Browser chooses 1–14 days
       |
POST /predict with {"horizon": 14}
       |
Pydantic validates request
       |
Load already-fitted model + historical observed target tail
       |
Forecast from saved cutoff: 2012-10-22
       |
Generate dates 2012-10-23 ... 2012-11-05
       |
Respond with numbers + model metadata + historical-demo warning
       |
Dashboard draws the line chart
```

**No fitting occurs during a normal API request.** The model is loaded from a persistent, pinned release artifact, and all requests forecast relative to that artifact's historical cutoff.

### Flow C: monitoring is a different simulation

```text
For each of 70 held-out test days:
   past truth through yesterday
       -> refit fixed model type (replay only)
       -> issue one-day prediction
       -> store issued forecast
       -> reveal today's truth
       -> store error
Daily errors -> rolling 28-day MAE -> flags -> SQLite / dashboard
```

**Do not confuse:** The monitor is a retrospective simulation of a daily-refit operating policy. The Docker-hosted API is a fixed historical model. Neither represents a live 2026 feed.

### Recommended reading order for the future repository

`README.md` → `docs/experiment_contract.md` → `forecastops/data/` → `forecastops/features/` → `forecastops/models/` → `forecastops/evaluation/` → `reports/selection.json` → `forecastops/monitoring/` → `forecastops/api/` → `web/` → `Dockerfile` → `tests/`.

---

## 5. Deployment and portfolio

### A. Deploy on Render Free using Docker

1. Finish the local pipeline, all tests, and `docker run` smoke checks.
2. Ensure GitHub contains the app, `Dockerfile`, pinned requirements, **release artifact**, required report summaries, and the **monitoring snapshot**. Keep scratch downloads and mutable local runtime DB out of version control. If artifact file sizes are unsuitable for Git, document a build-time artifact retrieval solution; avoid network training at startup.
3. Push the repository to GitHub. Confirm the intended application root (repo root or `ForecastOps/` subdirectory) matches where the `Dockerfile` lives.
4. In Render, choose **New → Web Service → connect the GitHub repository**.
5. Set service **Language/Runtime: Docker**, branch, and root directory if needed. Choose the **Free** instance type if offered.
6. Set `PORT=10000`, `FORECASTOPS_ARTIFACT_DIR`, `FORECASTOPS_SEED_DB`, `FORECASTOPS_DB_PATH`, and `FORECASTOPS_LOG_LEVEL=INFO` to the locations established for the Docker image. The application must bind to **`0.0.0.0:$PORT`**.
7. Configure the **health-check path `/health`** and deploy.
8. Visit your `.onrender.com` URL: confirm `/health` is ready; `/docs` loads; `POST /predict` returns 14 dated numerical predictions; `/monitoring` returns replay data; `/` shows charts and historical-mode disclosure.
9. Trigger a restart or redeploy and verify the app can reinitialize SQLite from its bundled snapshot. Treat any runtime writes as ephemeral, not permanently stored monitoring history.

**Free-tier limitations:** Render Free web services sleep after about **15 minutes of inactivity** and can require a cold start. Local changes, including SQLite files, disappear across spin-down/restarts/redeploys. The service is a **portfolio demo, not production infrastructure**. Check current limits before deployment: https://render.com/docs/free and https://render.com/docs/web-services.

### B. README outline

1. **ForecastOps**: one-sentence problem and a screenshot.
2. Why baselines matter; exactly what counts as beating seasonal naive.
3. Dataset and CC BY 4.0 attribution; historical 2011–2012 disclaimer.
4. Architecture diagram and folder map.
5. Chronological split table and five 14-day validation/test windows.
6. Models, features, and precautions against target leakage.
7. Real validation/test comparison table, per-horizon insight, and selected-model rationale.
8. Explain the **daily monitoring replay versus 14-day backtesting**.
9. Installation, pipeline run order, tests, API usage, Docker instructions.
10. Live demo + screenshots, limitations, and future enhancements.

### C. Screenshots to capture

- Dashboard top section with **historical cutoff**.
- 14-day forecast line chart with dated x-axis.
- Model comparison table showing test baseline improvement.
- Per-horizon or days 1–7 vs 8–14 accuracy chart.
- Historical monitor with rolling 28-day MAEs and flags.
- FastAPI `/docs` and a successful prediction response.
- Terminal with successful pytest and Docker smoke checks.

### D. Two-minute demonstration script

- **0:00–0:20:** Problem, data source, and why this is a historical demonstration.
- **0:20–0:45:** Chronological 591/70/70 split and five 14-day walk-forward windows.
- **0:45–1:10:** Show baselines versus candidate models and explain the frozen validation-only choice.
- **1:10–1:30:** Call `POST /predict` for 14 days; point out dates and model version.
- **1:30–1:50:** Explain separate 70-day day-ahead monitoring and 28-day rolling MAE.
- **1:50–2:00:** Docker, reproducibility, limitations; say clearly whether the complex model won.

### E. Resume bullets — replace placeholders only after measuring

- Built **ForecastOps**, a containerized **FastAPI** forecasting service for 1–14-day historical demand predictions using Python, scikit-learn, and statsmodels.
- Implemented **leakage-safe expanding-window backtesting** across five forecasting candidates, benchmarking against seasonal-naive demand forecasts and reporting validation/test MAE, RMSE, and WAPE; achieved **[measured result, if any]**.
- Developed an **idempotent SQLite-based historical monitoring replay** with 70 daily one-step-ahead forecasts and rolling 28-day error metrics; deployed an interactive demo using Docker and Render.

Never insert a fabricated accuracy percentage, latency result, or claim that a sophisticated model beat the baseline unless the output files prove it.

---

## Assumptions and intentional choices

The original source plan already specified the UCI daily dataset, 14-day horizon, five candidate models, 591/70/70 split, expanding five-window backtests, validation-only 1% gate, FastAPI, SQLite, and Render Free. This implementation guide retains those choices.

The following **clarifications/refinements** are intentional:

1. **Two distinct forecasting experiments:** non-overlapping **14-step backtests** for model selection/testing and an additional **daily 1-step replay** for monitoring. A 14-day fold does not produce 14 independently issued one-day forecasts.
2. **Refit policy in replay:** each day re-fit the same preselected model *type and fixed configuration* using only newly available past actuals. This is an offline operational simulation, **not** auto-retraining inside the hosted API.
3. **Historical serving artifact:** the default fixed model is trained **through 2012-10-22**, producing dated forecasts **starting 2012-10-23**. An optional post-evaluation full-data version would use a different cutoff and version.
4. **Stronger leakage tests:** mutate unseen future targets and assert predictions do not change; verify shifted rolling features, baseline recursion, and test/selection isolation.
5. **Metric resolution:** show per-fold and horizon-specific errors, in addition to pooled MAE/RMSE/WAPE. Aggregate only genuinely out-of-sample predictions.
6. **Monitoring edge cases:** support selected-is-baseline behavior, insufficient 28-day windows, baseline MAE zero, matching target dates, and repeat-run deduplication.
7. **Hyperparameters:** documented fixed starter defaults are included for reproducibility; they should be frozen before test evaluation.
8. **Development conventions:** editable Python package, `pip-tools`, explicit report/artifact directories, and offline build scripts are proposed implementation choices rather than requirements from the original brief.
9. **Time estimate:** increased from the original 24–36 hours to a more cautious 35–55 hours for a first complete implementation.
10. **No live-feed claim:** all plots, endpoints, README text, and CV statements identify the 2011–2012 data as historical unless a genuinely current data feed is implemented later.

### References

- UCI dataset page / license: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- UCI dataset citation: https://doi.org/10.24432/C5W894
- statsmodels Holt-Winters `ExponentialSmoothing`: https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
- Render web service guide: https://render.com/docs/web-services
- Render Free limits: https://render.com/docs/free

---

## Reusable prompt — create a walkthrough after implementation

> **Copy/paste the following into an LLM after the repository is actually implemented.**

```text
You are my senior ML engineer and a patient technical mentor. Inspect the ACTUAL implemented ForecastOps repository before writing. Do not infer file names, functions, model outcomes, architecture, or deployment behavior from this plan if the code differs.

Create ForecastOps/WALKTHROUGH.md with a beginner-to-advanced explanation that:
1. Starts with a simple explanation of forecasting, baselines, recursive multistep forecasts, chronological backtesting, and why leakage invalidates metrics.
2. Lists the real project tree and the best order to read files, with precise paths and key functions/classes.
3. Traces the dataset from day.csv through validation, features, models, folds, frozen selection, and untouched test evaluation, including actual dates and example shapes.
4. Traces ONE real POST /predict request through the Pydantic schema, route, model loading, history features, model inference, and actual JSON response. Distinguish fitting from inference.
5. Explains how validation-only selection works and quotes the genuine model comparison outputs. Say clearly if the selected model did not beat seasonal naive on held-out test data.
6. Separately traces the 70-day one-step-ahead historical monitoring replay: issue dates, target dates, the exact past-only refit policy, SQLite schemas, 28-day MAE, and flags. Explicitly contrast this with five 14-day test windows and with the fixed hosted API model.
7. Includes accurate Mermaid or ASCII diagrams and commands that REALLY run in the repository's current structure (PowerShell first).
8. Describes Docker packaging, Render configuration, the historical-data disclaimer, ephemeral SQLite behavior, and the limits of the demo.
9. Includes common bugs/debugging paths, test explanations, a glossary, and small exercises with expected results.
10. Ends with a 'Plan vs. actual implementation' table listing all deviations and reasons. Do not fabricate results, screenshots, or features; mark any missing implementation clearly.

Use friendly, straightforward language but retain technical correctness. Cite specific real repository files/functions throughout. Write only the walkthrough document; do not modify application code without approval.
```

# Verification and safe reproduction

Verified on **2026-11-05** using Python 3.13 locally and a Linux/amd64 Docker image. The five candidate configurations, chronological split, frozen Holt-Winters choice, and original release were preserved. The project is a local portfolio demo; no hosted deployment is claimed.

## Recorded results

| Check | Result |
|---|---|
| Automated suite | 45 passed; no skips |
| Dependencies | All 38 requirement pins match; `pip check` passes locally and in Docker |
| Frozen experiment | Original validation hashes checked strictly before isolated validation/test/release/replay reproduction |
| Prediction contract | Horizons 1/14, default 14, correct dates, finite nonnegative values, repeatability, strict invalid-input rejection |
| Readiness failures | Missing/corrupt model, manifest, selection, reports, seed, and runtime fixtures tested |
| Serving boundary | Readiness and prediction pass while fitting is forbidden |
| Monitoring replay | 140 issue rows / 70 actuals; reruns do not duplicate them; rolling metric starts on day 28 |
| Process recovery | Initial seed, existing-database restart, and fresh-database recovery return identical HTTP responses |
| Docker | Linux/amd64 build; default 10000/custom 18080 internal ports; UID 10001; writable runtime and read-only model; graceful restart and fresh-container recovery |
| Browser | Forecast changes, dated values, monitoring, and real Swagger requests checked; no unexpected captured console warnings/errors |
| Responsive layout | Widths 320/390/768/1280 checked without page-level overflow; comparison table scrolls internally |
| Preservation | Protected reports, data, fitted model, seed and existing runtime bytes unchanged |
| Hosting | Not deployed |

These are development verification results, not a claim that CI runs on every push. Screenshots in [screenshots/](screenshots/) show the verified Docker-served interface. Raw command logs, temporary databases, build metadata and reproduction outputs remain under ignored `output/verification/`.

## Run the included release

Use the Docker commands in the [README](../README.md#run-with-docker), or start the pinned Python environment:

```powershell
.\.venv\Scripts\python.exe -m uvicorn forecastops.api.main:app --host 127.0.0.1 --port 8000
```

No fitting is needed. The first readiness/monitoring request copies the bundled seed when the runtime DB is absent. An existing corrupt database returns 503 and is retained for inspection; choose a new `FORECASTOPS_DB_PATH` to recover the snapshot without deleting existing state. `.env.example` is explanatory and is not loaded automatically.

`/health` checks release/selection provenance, inference, runtime replay contents, report shape/numerical fields, and required dashboard assets. A successful HTML response from `/` alone does not establish readiness.

## Repeat checks safely

Install the pinned requirements first. Each output directory below must be new:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -o faulthandler_timeout=60
.\.venv\Scripts\python.exe -m scripts.smoke_server --output-dir output/verification/new-http-run --port 18765
.\.venv\Scripts\python.exe -m scripts.verify_docker --image forecastops:local --output-dir output/verification/new-docker-run
.\.venv\Scripts\python.exe -m scripts.verify_experiment --output-dir output/verification/new-experiment-run
```

The Docker helper expects an image built beforehand. It manages only containers created by that run, publishes on loopback, checks HTTP behavior, permissions, package pins, copied bytes, local/Linux inference, restart and fresh-container recovery, then removes its containers. `--keep-containers` supports a browser follow-up; saved IDs identify exactly what to clean up afterward. Ephemeral published host ports can change on restart, so the helper refreshes their mappings. Use an explicit host port for a stable manual demo URL.

`verify_experiment` refits fixed candidates only into an isolated output tree. It checks the original raw checksum, raw/processed equality and strict frozen validation provenance. Reproduced validation is compared before test evaluation. Each regenerated selection/manifest binds its own bytes. Original artifacts and model choice are never replaced.

Numerical comparisons retain row/schema/date/value/metric/status/count fields, excluding only measured timing columns (including validation/test suffixes) and creation/generation timestamps. Tolerance is `rtol=1e-10`, `atol=1e-7`. The original frozen byte hashes remain strict provenance checks, separate from tolerant numerical reproduction.

Tests use temporary databases and assert that protected local files remain unchanged. Replay writes a summary only when explicitly requested. Legacy generation commands refuse overwriting frozen output; do not run old pipeline creation commands against `reports/`.

## Troubleshooting

- **TestClient stalls before a route runs:** on restricted Windows runners, asyncio may block creating its loopback socketpair. The bounded `python -m scripts.diagnose_api_test --label new-check` probes distinguish this from an app failure. Use a local shell/runner permitting loopback; dependencies and route types do not need to change.
- **Docker command or credential helper is missing after installation:** open a new shell, or add the installed Docker tools directory to the current process's PATH. `verify_docker --docker FULL_PATH` accepts an explicit CLI executable.
- **Container readiness fails:** inspect container logs and `/health`; confirm the artifact/report/seed files are present and the runtime directory is writable. Do not regenerate the model to hide a packaging issue.
- **Blank early monitoring region:** the first 27 days have no full rolling window. The first 28-day MAE is dated 2012-11-19.
- **Ridge has lower test error:** retain the validation-selected Holt-Winters model. Held-out results cannot justify reselection.

## Reproducibility limits

Docker startup uses `exec uvicorn`, allowing graceful PID-1 shutdown. The measured image uses a floating `python:3.13-slim` base tag; future rebuilds may resolve a different base digest. Pinned Python dependencies do not guarantee a byte-identical future image. Docker checks establish local Linux container behavior, not public hosting availability or provider persistence. The monitoring replay is historical and the API does not ingest new demand data.

# Deploy ForecastOps for an interview demo

Last checked: 2026-10-28. This describes a **historical portfolio demo**, not a live forecasting service. The model artifact forecasts dates after 2012-10-22.

## Recommended host: Render Free web service

Render is the simplest fit for this repository. It [builds Docker images from a repository Dockerfile](https://render.com/docs/docker), gives a public `onrender.com` URL, and supports an HTTP [health check](https://render.com/docs/health-checks). The `Dockerfile` is at the repository root, installs pinned dependencies, copies the prebuilt model and reports, and starts Uvicorn on `0.0.0.0:${PORT:-10000}`. Render builds the image remotely, so you can create the service even if Docker is not installed on your laptop. **That remote build does not replace the pending local Docker smoke test.**

### Setup in the Render dashboard

1. Push the complete repository to `https://github.com/saidarshan20/ForecastOps.git` on branch `main`. Confirm GitHub shows `Dockerfile`, `requirements.txt`, `artifacts/release/model.joblib`, `artifacts/release/manifest.json`, `artifacts/monitoring_seed.sqlite`, `reports/`, and `web/`.
2. Sign in to [Render](https://dashboard.render.com/) and connect the GitHub account. Choose **New → Web Service → Git Provider → saidarshan20/ForecastOps**. A linked provider supports automatic deploys on later pushes; a public repository URL is an alternative but [does not support automatic deploys](https://render.com/docs/web-services).
3. Select branch **main**, language/runtime **Docker**, Dockerfile path **`Dockerfile`**, and repository root as the root directory. Select the **Free** instance if it is offered for your account.
4. Set the HTTP health-check path to **`/health`**. The Dockerfile already sets `PORT=10000` and the artifact/database paths; you can leave these defaults unless your Render configuration requires overrides. Do not set a start command that runs training scripts.
5. Create the service. Watch the build and startup logs. The service should load prebuilt artifacts and answer `/health` with `{"status":"ready","mode":"historical_demo"}`.
6. Open the assigned `https://<name>.onrender.com/` URL, then check `/docs`, `/model`, and `/monitoring`. In `/docs`, execute `POST /predict` with `{"horizon":14}`. Check for exactly 14 numeric predictions, from **2012-10-23** to **2012-11-05**, and the historical disclaimer.
7. Put the verified live URL in the root README and on your resume/portfolio. Before an interview, open the page once so a sleeping Free instance can wake.

Render's [Free plan documentation](https://render.com/docs/free) says Free web services spin down after 15 minutes without inbound traffic, may take about a minute to wake, and lose local filesystem changes when they spin down or redeploy. This app's SQLite file is seeded from the committed snapshot when missing, so the historical replay can be restored. The service should not be presented as storing new monitoring data permanently. Render's terms can change; confirm the plan shown in your account before creating the service.

### Checks after each deploy

```powershell
$base = 'https://YOUR-SERVICE.onrender.com'
Invoke-RestMethod "$base/health"
Invoke-RestMethod "$base/model"
Invoke-RestMethod "$base/monitoring"
Invoke-RestMethod -Uri "$base/predict" -Method Post -ContentType 'application/json' -Body '{"horizon":14}'
```

Expected: `/health` ready; `/model` cutoff `2012-10-22`; `/monitoring` mode `historical_replay` and 70 paired days; `/predict` 14 nonnegative numbers on historical dates. Visit `/` and `/docs` in a browser too. If `/health` returns 503, inspect Render logs for a missing artifact, model load error, or unwritable runtime path. If the service never becomes reachable, confirm the Docker process binds `0.0.0.0` on `PORT` and that the health-check path is `/health`.

## Other hosts

| Host | Fit for this repo | Tradeoff |
|---|---|---|
| **Render Free** | Recommended for a first interview link and existing Dockerfile | Sleeps after idle; ephemeral SQLite runtime file. [Render Free docs](https://render.com/docs/free) |
| **Railway Free** | Can run a Docker-based web service | Current Free plan includes **$1 monthly credit** after the trial; check usage and account eligibility. [Railway plans](https://docs.railway.com/pricing/plans), [trial](https://docs.railway.com/pricing/free-trial) |
| **Hugging Face Spaces** | Familiar ML portfolio audience, supports Docker Spaces | Its current documentation says creating compute Spaces requires a paid plan; verify account eligibility and cost before choosing it. [Spaces overview](https://huggingface.co/docs/hub/spaces-overview) |

For an interview, put the **GitHub repository**, **live dashboard URL**, and **a screenshot** together. Explain that the model was selected using validation only, that the 14-day backtest differs from the one-day monitoring replay, and that the hosted model is fixed at a 2012 cutoff. Do not describe it as a 2026 operational forecast.

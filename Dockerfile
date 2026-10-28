FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY forecastops ./forecastops
COPY web ./web
COPY artifacts ./artifacts
COPY reports ./reports
RUN mkdir -p /app/runtime && useradd -r -u 10001 forecastops && chown -R forecastops:forecastops /app/runtime
USER forecastops
ENV FORECASTOPS_ARTIFACT_DIR=/app/artifacts/release FORECASTOPS_DB_PATH=/app/runtime/monitoring.sqlite FORECASTOPS_SEED_DB=/app/artifacts/monitoring_seed.sqlite PORT=10000
EXPOSE 10000
CMD ["sh", "-c", "uvicorn forecastops.api.main:app --host 0.0.0.0 --port ${PORT:-10000}"]

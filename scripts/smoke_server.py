"""Bounded real HTTP checks, including existing-DB restart and fresh-seed recovery."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import httpx


def check(base):
    result = {}
    with httpx.Client(base_url=base, timeout=10, trust_env=False) as client:
        for route in ['/health', '/model', '/monitoring', '/comparison', '/horizons', '/docs', '/', '/static/app.js', '/static/styles.css', '/openapi.json']:
            r = client.get(route)
            assert r.status_code == 200, (route, r.status_code, r.text)
            result[route] = r.json() if 'application/json' in r.headers.get('content-type', '') else {'status': r.status_code, 'bytes': len(r.content)}
        for horizon in [1, 14]:
            r = client.post('/predict', json={'horizon': horizon})
            assert r.status_code == 200
            body = r.json()
            assert len(body['predictions']) == horizon
            assert body['forecast_start'] == '2012-10-23'
            assert body['forecast_end'] == ('2012-10-23' if horizon == 1 else '2012-11-05')
            assert body['trained_through'] == '2012-10-22'
            assert body['model_version'] == 'v1-20121022'
            assert all(math.isfinite(p['value']) and p['value'] >= 0 for p in body['predictions'])
            assert client.post('/predict', json={'horizon': horizon}).json() == body
            result[f'/predict?horizon={horizon}'] = body
        for invalid in [0, 15, 1.5, '7', True, None, [], {}]:
            assert client.post('/predict', json={'horizon': invalid}).status_code == 422
        for token in ['NaN', 'Infinity', '-Infinity']:
            assert client.post('/predict', content='{"horizon":' + token + '}', headers={'content-type': 'application/json'}).status_code == 422
        assert client.post('/predict', json={'horizon': 14, 'extra': 1}).status_code == 422
        assert result['/monitoring']['paired_days'] == 70
        assert result['/monitoring']['records'][26]['selected_rolling_28_mae'] is None
        assert result['/monitoring']['records'][27]['selected_rolling_28_mae'] is not None
    return result


def main(output, port):
    output.mkdir(parents=True, exist_ok=False)
    observations = []
    for name, database in [('initial', output / 'runtime.sqlite'), ('restart', output / 'runtime.sqlite'), ('fresh', output / 'fresh.sqlite')]:
        env = {**os.environ, 'FORECASTOPS_DB_PATH': str(database.resolve())}
        with (output / f'{name}.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'forecastops.api.main:app', '--host', '127.0.0.1', '--port', str(port)], env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 30
                while True:
                    try:
                        with httpx.Client(timeout=2, trust_env=False) as client:
                            r = client.get(f'http://127.0.0.1:{port}/health')
                            r.raise_for_status()
                        break
                    except (httpx.HTTPError, OSError):
                        if time.monotonic() > deadline or process.poll() is not None:
                            raise RuntimeError(f'{name} startup failed; inspect log')
                        time.sleep(.2)
                result = check(f'http://127.0.0.1:{port}')
                observations.append(result)
                (output / f'{name}.json').write_text(json.dumps(result, indent=2) + '\n')
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        if name == 'initial':
            original_db_hash = hashlib.sha256(database.read_bytes()).hexdigest()
        if name == 'restart':
            assert hashlib.sha256(database.read_bytes()).hexdigest() == original_db_hash
    assert observations[0] == observations[1] == observations[2]
    (output / 'result.json').write_text(json.dumps({'status': 'passed', 'port': port, 'phases': ['initial seed', 'existing database restart (byte unchanged)', 'fresh database seed recovery'], 'identical_endpoint_responses': True}, indent=2) + '\n')
    print('Real HTTP checks passed in all three server processes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18765)
    args = parser.parse_args()
    main(args.output_dir, args.port)

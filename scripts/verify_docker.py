"""Verify a prebuilt image without fitting or changing frozen/local runtime files.

Build separately, then pass --image and a new --output-dir. Only containers
created by this run are managed. --keep-containers permits a browser follow-up.
"""
import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import uuid

import httpx
import joblib
import numpy as np

from forecastops.config import ROOT
from scripts.smoke_server import check


def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ['reports', 'artifacts', 'runtime', 'data/processed']
            for p in (ROOT / folder).rglob('*') if p.is_file()}


def verify(docker, image, output, keep):
    output.mkdir(parents=True, exist_ok=False)
    before = hashes()
    (output / 'protected-before.json').write_text(json.dumps(before, indent=2))
    containers, commands = [], []
    result = {'status': 'running', 'image': image, 'containers': containers}

    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

    def command(*args, timeout=60):
        completed = subprocess.run([docker, *args], capture_output=True, text=True,
                                   encoding='utf-8', errors='replace', timeout=timeout)
        commands.append({'args': list(args), 'returncode': completed.returncode,
                         'stdout': completed.stdout, 'stderr': completed.stderr})
        save('commands.json', commands)
        if completed.returncode:
            raise RuntimeError(f'Docker {args[0]} failed: {completed.stderr}')
        return (completed.stdout + completed.stderr if args[0] == 'logs' else completed.stdout).strip()

    def probe(container, code):
        return json.loads(command('exec', container['id'], 'python', '-c', code))

    def ready(container):
        deadline = time.monotonic() + 60
        with httpx.Client(timeout=3, trust_env=False) as client:
            while time.monotonic() < deadline:
                try:
                    response = client.get(container['url'] + '/health')
                    if response.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(.25)
        raise RuntimeError(f"Container not ready: {container['name']}")

    def address(container):
        # Docker may reassign an ephemeral published host port on restart.
        inspection = json.loads(command('inspect', container['id']))[0]
        binding = inspection['NetworkSettings']['Ports'][f"{container['internal_port']}/tcp"][0]
        assert binding['HostIp'] == '127.0.0.1'
        container['url'] = f"http://127.0.0.1:{binding['HostPort']}"
        container.setdefault('url_history', []).append(container['url'])
        return inspection

    def start(label, port, custom=False):
        name = f'forecastops-verify-{uuid.uuid4().hex[:10]}-{label}'
        args = ['run', '--detach', '--name', name, '--publish', f'127.0.0.1::{port}']
        if custom:
            args.extend(['--env', f'PORT={port}'])
        cid = command(*args, image)
        container = {'id': cid, 'name': name, 'internal_port': port}
        containers.append(container)
        inspection = address(container)
        save(label + '-inspect.json', inspection)
        assert not inspection['Mounts'], 'Fresh verification containers must have no volumes'
        # Observe a genuinely missing DB before the first readiness request seeds it.
        assert probe(container, "import json; from pathlib import Path; print(json.dumps(Path('/app/runtime/monitoring.sqlite').exists()))") is False
        ready(container)
        print(f"Ready: {label} at {container['url']}", flush=True)
        return container

    try:
        save('docker-version.json', json.loads(command('version', '--format', '{{json .}}')))
        metadata = json.loads(command('image', 'inspect', image))[0]
        save('image.json', metadata)
        assert metadata['Os'] == 'linux'
        assert metadata['Config']['User'] == 'forecastops'
        result.update(image_id=metadata['Id'], platform=f"{metadata['Os']}/{metadata['Architecture']}", image_bytes=metadata['Size'])
        first = start('initial', 10000)
        pid1 = probe(first, "import json; from pathlib import Path; print(json.dumps(Path('/proc/1/cmdline').read_bytes().decode().strip('\\0').split('\\0')))")
        assert 'uvicorn' in ' '.join(pid1) and not pid1[0].endswith('sh')
        save('pid1.json', pid1)
        initial = check(first['url'])
        save('initial-http.json', initial)

        runtime_code = """import json,os,hashlib,sqlite3
from pathlib import Path
p=Path('/app/runtime/monitoring.sqlite')
with sqlite3.connect(p.as_uri()+'?mode=ro',uri=True) as db:
 counts={t:db.execute('SELECT count(*) FROM '+t).fetchone()[0] for t in ['forecast_issues','actuals','monitoring_runs']}
 integrity=db.execute('PRAGMA integrity_check').fetchone()[0]
print(json.dumps({'uid':os.getuid(),'runtime_writable':os.access(p.parent,os.W_OK),'release_writable':os.access('/app/artifacts/release/model.joblib',os.W_OK),'db_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'counts':counts,'integrity':integrity}))"""
        initial_state = probe(first, runtime_code)
        save('initial-runtime.json', initial_state)
        assert initial_state['uid'] == 10001
        assert initial_state['runtime_writable'] and not initial_state['release_writable']
        assert initial_state['counts'] == {'forecast_issues': 140, 'actuals': 70, 'monitoring_runs': 1}
        assert initial_state['integrity'] == 'ok'
        assert initial_state['db_sha256'] == hashlib.sha256((ROOT / 'artifacts/monitoring_seed.sqlite').read_bytes()).hexdigest()

        pins = dict(re.findall(r'^([A-Za-z0-9_.-]+)==([^\s]+)', (ROOT / 'requirements.txt').read_text(), re.M))
        installed = probe(first, 'import json,importlib.metadata as m; print(json.dumps({name:m.version(name) for name in ' + repr(list(pins)) + '}))')
        assert installed == pins
        save('installed-pins.json', installed)
        command('exec', first['id'], 'python', '-m', 'pip', 'check')

        # Check copied release/report/static bytes and exclusions inside the image.
        expected = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                    for folder in ['reports', 'artifacts', 'web', 'forecastops']
                    for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        code = "import json,hashlib; from pathlib import Path; names=" + repr(list(expected)) + "; print(json.dumps({name:hashlib.sha256((Path('/app')/name).read_bytes()).hexdigest() for name in names}))"
        assert probe(first, code) == expected
        excluded = ['.git', '.env', 'datasets', 'docs', 'output', 'scripts', 'tests', 'TASK_TRACKER.md', 'WALKTHROUGH.md', 'ForecastOps_Implementation_Plan.md']
        assert probe(first, "import json; from pathlib import Path; print(json.dumps([x for x in " + repr(excluded) + " if (Path('/app')/x).exists()]))") == []
        save('image-input-hashes.json', expected)

        # Compare Linux inference against the original fitted Windows payload;
        # no fit, re-selection, or held-out-result tuning occurs here.
        payload = joblib.load(ROOT / 'artifacts/release/model.joblib')
        expected_values = payload['model'].predict(14, payload['history'].index[-1])
        actual_values = [p['value'] for p in initial['/predict?horizon=14']['predictions']]
        np.testing.assert_allclose(actual_values, expected_values, rtol=1e-10, atol=1e-7)
        for horizon in [1, 14]:
            points = initial[f'/predict?horizon={horizon}']['predictions']
            assert [p['date'] for p in points] == [(date(2012, 10, 23) + timedelta(days=i)).isoformat() for i in range(horizon)]

        # Serving remains functional when fitting is forbidden, inside Linux too.
        nofit = probe(first, """import json
from forecastops.models.statistical import HoltWinters
def forbidden(*args,**kwargs): raise AssertionError('Serving must never fit')
HoltWinters.fit=forbidden
from forecastops.api.main import health,make_prediction
from forecastops.api.schemas import PredictRequest
print(json.dumps({'health':health(),'points':len(make_prediction(PredictRequest(horizon=14))['predictions'])}))""")
        assert nofit['points'] == 14 and nofit['health']['status'] == 'ready'
        save('no-fitting.json', nofit)

        command('restart', '--timeout', '10', first['id'], timeout=45)
        save('restart-inspect.json', address(first))
        ready(first)
        assert 'Finished server process [1]' in command('logs', first['id']), 'Restart must shut Uvicorn down gracefully'
        restarted = check(first['url'])
        save('restart-http.json', restarted)
        restarted_state = probe(first, runtime_code)
        save('restart-runtime.json', restarted_state)
        assert restarted == initial
        assert restarted_state == initial_state
        print('Same-container restart passed; runtime bytes unchanged.', flush=True)

        fresh = start('fresh-custom-port', 18080, custom=True)
        fresh_result = check(fresh['url'])
        save('fresh-http.json', fresh_result)
        fresh_state = probe(fresh, runtime_code)
        save('fresh-runtime.json', fresh_state)
        assert fresh_result == initial and fresh_state == initial_state
        # Confirm the overridden process is not listening on the old internal port.
        assert probe(fresh, "import json,socket; s=socket.socket(); s.settimeout(1); print(json.dumps(s.connect_ex(('127.0.0.1',10000)) != 0)); s.close()") is True
        print('Fresh-container seeding and custom internal PORT=18080 passed.', flush=True)
        result.update(status='passed', pinned_packages=len(pins), identical_http_responses=True,
                      restart_database_unchanged=True, fresh_database_from_seed=True,
                      local_prediction_match={'rtol': 1e-10, 'atol': 1e-7},
                      nonroot_uid=initial_state['uid'], image_bytes_match=True,
                      no_runtime_fitting=True, keep_containers=keep)
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        for container in containers:
            try:
                (output / f"{container['name']}.log").write_text(command('logs', container['id']), encoding='utf-8')
            except Exception as exc:
                result.setdefault('log_errors', []).append(str(exc))
            if not keep or result['status'] != 'passed':
                try:
                    command('stop', '--timeout', '10', container['id'], timeout=30)
                    command('rm', container['id'])
                except Exception as exc:
                    result.setdefault('cleanup_errors', []).append(str(exc))
        after = hashes()
        save('protected-after.json', after)
        result['protected_files'] = len(before)
        result['local_evidence_unchanged'] = before == after
        if before != after:
            result.update(status='failed', preservation_error='Protected local files changed')
        save('result.json', result)
        if before != after:
            raise AssertionError('Protected local files changed')
    print(f"Docker verification passed. Evidence: {output}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docker', default='docker')
    parser.add_argument('--image', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--keep-containers', action='store_true')
    args = parser.parse_args()
    verify(args.docker, args.image, args.output_dir.resolve(), args.keep_containers)

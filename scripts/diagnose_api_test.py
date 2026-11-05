"""Bounded, read-only diagnostics for the TestClient transport."""
import subprocess
import sys
from pathlib import Path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='default')
    args = parser.parse_args()
    output = Path('output/verification/diagnostics') / args.label
    output.mkdir(parents=True, exist_ok=True)
    probes = {
        'minimal': '''import faulthandler
faulthandler.dump_traceback_later(10, exit=True)
from fastapi import FastAPI
from fastapi.testclient import TestClient
app = FastAPI()
@app.get('/')
def root(): return {'ok': True}
print(TestClient(app).get('/'), flush=True)
''',
        'async_route': '''import faulthandler
faulthandler.dump_traceback_later(10, exit=True)
from fastapi import FastAPI
from fastapi.testclient import TestClient
app = FastAPI()
@app.get('/')
async def root(): return {'ok': True}
print(TestClient(app).get('/'), flush=True)
''',
        'direct_release': '''import faulthandler
faulthandler.dump_traceback_later(10, exit=True)
from forecastops.api.service import load_release, predict
m, p = load_release()
print(predict(m, p, 1), flush=True)
''',
    }
    for name, code in probes.items():
        try:
            result = subprocess.run([sys.executable, '-u', '-c', code], capture_output=True, text=True, timeout=20)
            text = f'exit={result.returncode}\n{result.stdout}\n{result.stderr}'
        except subprocess.TimeoutExpired as exc:
            text = f'timeout\n{exc.stdout!r}\n{exc.stderr!r}'
        (output / f'{name}.txt').write_text(text, encoding='utf-8')
        print(name, text, flush=True)


if __name__ == '__main__':
    main()

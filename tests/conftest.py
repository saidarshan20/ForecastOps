"""Tests may read frozen evidence but never mutate it or existing runtime state."""
import hashlib
from pathlib import Path
import pytest
from forecastops.config import ROOT


def protected_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for name in ["reports", "artifacts", "runtime", "data/processed"]
            for p in (ROOT / name).rglob("*") if p.is_file()}


@pytest.fixture(scope="session", autouse=True)
def preserve_evidence():
    before = protected_hashes()
    yield
    assert protected_hashes() == before, "Tests changed protected experiment/runtime files"


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch, tmp_path):
    from forecastops.api import service
    monkeypatch.setattr(service, "DB_PATH", tmp_path / "runtime/monitoring.sqlite")

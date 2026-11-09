"""Shared test isolation."""
import pytest


@pytest.fixture(autouse=True)
def no_pool_deadline(monkeypatch):
    """The PBS pool exports its deadline to every task, including the infrastructure
    job's pytest; tests must never defer or write the real pool's defer marker."""
    for name in ("RCG_DEADLINE", "RCG_DEFER_FILE", "RCG_EVAL_ESTIMATE_S"):
        monkeypatch.delenv(name, raising=False)

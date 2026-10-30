"""Regression tests for trace updates used by the dashboard."""

import asyncio
import sqlite3
import tempfile

from backend.app import observe
from backend.app.routes.v1.stats import v1_stats_dashboard


def test_trace_updates_keep_search_results_and_dashboard_rate(monkeypatch):
    path = tempfile.mktemp(suffix=".db")
    monkeypatch.setattr(observe, "_DB_PATH", path)
    monkeypatch.setattr(observe, "_initialized", False)

    observe.write_trace("search-1", mode="search", n_results=7, question="scene")
    observe.write_trace("search-1", endpoint="/api/v1/search", status="200", latency_ms=12)
    observe.write_trace("chat-1", mode="chat", status="200")

    with sqlite3.connect(path) as db:
        assert db.execute(
            "SELECT mode, n_results, query_raw FROM traces WHERE trace_id = 'search-1'"
        ).fetchone() == ("search", 7, "scene")
    db.close()

    dashboard = asyncio.run(v1_stats_dashboard())
    assert dashboard["retrieval"]["zero_results"] == 0
    assert dashboard["retrieval"]["zero_rate"] == 0

"""v1 stats endpoints: /api/v1/stats/latency, /api/v1/stats/dashboard."""

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Query

from ...observe import _db_path, ensure_db

logger = logging.getLogger(__name__)

router = APIRouter()


def _percentile(sorted_list: list[float], pct: float) -> float:
    if not sorted_list:
        return 0.0
    idx = int(len(sorted_list) * pct)
    idx = min(idx, len(sorted_list) - 1)
    return sorted_list[idx]


def _safe_rate(numerator: int, denominator: int) -> float:
    """Return percentage (0-100) or 0.0 if denominator is zero."""
    if denominator == 0:
        return 0.0
    return round(numerator / denominator * 100, 2)


@router.get("/stats/latency")
async def v1_stats_latency(limit: int = Query(100, ge=10, le=1000)):
    """Latency percentiles by span name over recent traces."""
    ensure_db()
    with sqlite3.connect(_db_path(), timeout=5) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute("""
            SELECT s.name, s.latency_ms
            FROM spans s
            JOIN (SELECT trace_id FROM traces ORDER BY ts DESC LIMIT ?) t
                ON s.trace_id = t.trace_id
            WHERE s.latency_ms IS NOT NULL
            ORDER BY s.name, s.latency_ms
        """, (limit,)).fetchall()

    by_name: dict[str, list[float]] = {}
    for r in rows:
        by_name.setdefault(r["name"], []).append(r["latency_ms"])

    stats = {}
    for name, latencies in sorted(by_name.items()):
        n = len(latencies)
        latencies.sort()
        stats[name] = {
            "count": n,
            "p50": latencies[n // 2] if n else 0,
            "p95": latencies[int(n * 0.95)] if n else 0,
            "max": latencies[-1] if n else 0,
        }

    return {"stats": stats, "n_traces": limit}


@router.get("/stats/dashboard")
async def v1_stats_dashboard():
    """Aggregated dashboard statistics — all panel data in one response."""
    ensure_db()
    now = time.time()
    day_ago = now - 86400

    with sqlite3.connect(_db_path(), timeout=5) as c:
        c.row_factory = sqlite3.Row

        # ── Operations ──────────────────────────────────────────────
        n_requests = c.execute(
            "SELECT COUNT(*) as cnt FROM traces WHERE ts > ? AND COALESCE(endpoint, '') NOT LIKE '/api/v1/stats/%'", (day_ago,)
        ).fetchone()["cnt"]

        n_errors = c.execute(
            "SELECT COUNT(*) as cnt FROM traces WHERE (status = 'error' OR CAST(status AS INTEGER) >= 400) AND ts > ? AND COALESCE(endpoint, '') NOT LIKE '/api/v1/stats/%'",
            (day_ago,),
        ).fetchone()["cnt"]

        n_timeouts = c.execute(
            "SELECT COUNT(*) as cnt FROM traces WHERE error_label = 'timeout' AND ts > ? AND COALESCE(endpoint, '') NOT LIKE '/api/v1/stats/%'",
            (day_ago,),
        ).fetchone()["cnt"]

        n_fallbacks = c.execute(
            "SELECT COUNT(*) as cnt FROM traces WHERE mode = 'fallback' AND ts > ? AND COALESCE(endpoint, '') NOT LIKE '/api/v1/stats/%'",
            (day_ago,),
        ).fetchone()["cnt"]

        # QPS hourly buckets (last 24h)
        qps_rows = c.execute("""
            SELECT ts FROM traces WHERE ts > ? AND COALESCE(endpoint, '') NOT LIKE '/api/v1/stats/%'
        """, (day_ago,)).fetchall()

        hourly_buckets: dict[str, int] = {}
        for r in qps_rows:
            hour_key = datetime.fromtimestamp(r["ts"], tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:00"
            )
            hourly_buckets[hour_key] = hourly_buckets.get(hour_key, 0) + 1

        qps_hourly = [
            {"hour": h, "count": cnt}
            for h, cnt in sorted(hourly_buckets.items())
        ]

        # ── Retrieval Quality ───────────────────────────────────────
        n_zero = c.execute(
            "SELECT COUNT(*) as cnt FROM traces WHERE n_results = 0 AND mode IN ('search', 'search_parsed', 'image_search') AND ts > ?",
            (day_ago,),
        ).fetchone()["cnt"]
        n_searches = c.execute(
            "SELECT COUNT(*) as cnt FROM traces WHERE mode IN ('search', 'search_parsed', 'image_search') AND ts > ?",
            (day_ago,),
        ).fetchone()["cnt"]

        # Source contribution from span attrs (best-effort)
        source_contribution: dict[str, int] = {}
        try:
            src_rows = c.execute("""
                SELECT s.attrs FROM spans s
                JOIN traces t ON s.trace_id = t.trace_id
                WHERE t.ts > ? AND s.name LIKE 'search.%' AND s.attrs IS NOT NULL
            """, (day_ago,)).fetchall()
            for r in src_rows:
                try:
                    attrs = json.loads(r["attrs"])
                    src = attrs.get("source") or attrs.get("retriever")
                    if src:
                        source_contribution[src] = source_contribution.get(src, 0) + 1
                except (json.JSONDecodeError, TypeError):
                    pass
        except Exception:
            logger.debug("source_contribution extraction failed", exc_info=True)

        # ── Answer Quality ──────────────────────────────────────────
        answer_rows = c.execute(
            "SELECT status, COUNT(*) as cnt FROM answers WHERE ts > ? GROUP BY status",
            (day_ago,),
        ).fetchall()
        answer_status = {r["status"]: r["cnt"] for r in answer_rows}

        # Evidence rate
        total_answers = c.execute(
            "SELECT COUNT(*) as cnt FROM answers WHERE ts > ?", (day_ago,)
        ).fetchone()["cnt"]
        with_evidence = c.execute(
            "SELECT COUNT(*) as cnt FROM answers WHERE ts > ? AND evidence IS NOT NULL AND evidence != '' AND evidence != '[]'",
            (day_ago,),
        ).fetchone()["cnt"]
        evidence_rate = _safe_rate(with_evidence, total_answers)

        # Feedback
        feedback_rows = c.execute(
            "SELECT rating, COUNT(*) as cnt FROM feedback WHERE ts > ? GROUP BY rating",
            (day_ago,),
        ).fetchall()
        feedback = {str(r["rating"]): r["cnt"] for r in feedback_rows}

        # Feedback hourly
        fb_ts_rows = c.execute(
            "SELECT rating, ts FROM feedback WHERE ts > ?", (day_ago,)
        ).fetchall()
        fb_hourly: dict[str, dict[str, int]] = {}
        for r in fb_ts_rows:
            hour_key = datetime.fromtimestamp(r["ts"], tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:00"
            )
            bucket = fb_hourly.setdefault(hour_key, {"up": 0, "down": 0})
            if r["rating"] == 1:
                bucket["up"] += 1
            elif r["rating"] == -1:
                bucket["down"] += 1
        feedback_hourly = [
            {"hour": h, "up": v["up"], "down": v["down"]}
            for h, v in sorted(fb_hourly.items())
        ]

        # ── Cost ────────────────────────────────────────────────────
        cost_row = c.execute("""
            SELECT
                COALESCE(SUM(tokens_in), 0) as total_in,
                COALESCE(SUM(tokens_out), 0) as total_out,
                COALESCE(AVG(NULLIF(tokens_in, 0)), 0) as avg_in,
                COALESCE(AVG(NULLIF(tokens_out, 0)), 0) as avg_out
            FROM traces WHERE ts > ?
        """, (day_ago,)).fetchone()

        # ── Agent ───────────────────────────────────────────────────
        agent_spans = c.execute("""
            SELECT s.name, s.error, s.attrs FROM spans s
            JOIN traces t ON s.trace_id = t.trace_id
            WHERE t.ts > ? AND (s.name LIKE 'agent.%' OR s.name LIKE 'tool.%')
        """, (day_ago,)).fetchall()

        agent_total = 0
        agent_steps: list[int] = []
        tool_dist: dict[str, int] = {}
        tool_errors = 0
        max_steps_hit = 0

        # Group agent spans by trace to count steps per run
        agent_traces: dict[str, int] = {}
        for r in agent_spans:
            name = r["name"]
            if name.startswith("agent."):
                agent_traces[name] = agent_traces.get(name, 0) + 1
            if name.startswith("tool."):
                tool_name = name.split(".", 1)[1] if "." in name else name
                tool_dist[tool_name] = tool_dist.get(tool_name, 0) + 1
                if r["error"]:
                    tool_errors += 1

        # Count agent runs from spans grouped by trace_id
        agent_run_rows = c.execute("""
            SELECT s.trace_id, COUNT(*) as steps FROM spans s
            JOIN traces t ON s.trace_id = t.trace_id
            WHERE t.ts > ? AND s.name LIKE 'agent.step%'
            GROUP BY s.trace_id
        """, (day_ago,)).fetchall()
        agent_total = len(agent_run_rows)
        for r in agent_run_rows:
            agent_steps.append(r["steps"])
            # Check if max steps was hit (heuristic: attrs might contain it)

        avg_steps = round(sum(agent_steps) / len(agent_steps), 1) if agent_steps else 0.0

        # Max steps hit: check for spans with name like 'agent.max_steps'
        max_steps_hit = c.execute("""
            SELECT COUNT(*) as cnt FROM spans s
            JOIN traces t ON s.trace_id = t.trace_id
            WHERE t.ts > ? AND s.name = 'agent.max_steps'
        """, (day_ago,)).fetchone()["cnt"]

        # ── Data & Index ────────────────────────────────────────────
        index_row = c.execute(
            "SELECT index_version, encoder, reranker, prompt_version, llm_model, vlm_model "
            "FROM traces ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        index_info = dict(index_row) if index_row else {}

        # ── Latency (inline) ────────────────────────────────────────
        lat_rows = c.execute("""
            SELECT s.name, s.latency_ms
            FROM spans s
            JOIN (SELECT trace_id FROM traces WHERE ts > ? ORDER BY ts DESC LIMIT 200) t
                ON s.trace_id = t.trace_id
            WHERE s.latency_ms IS NOT NULL
            ORDER BY s.name, s.latency_ms
        """, (day_ago,)).fetchall()

        lat_by_name: dict[str, list[float]] = {}
        for r in lat_rows:
            lat_by_name.setdefault(r["name"], []).append(r["latency_ms"])

        latency = {}
        for name, lats in sorted(lat_by_name.items()):
            lats.sort()
            n = len(lats)
            latency[name] = {
                "count": n,
                "p50": round(_percentile(lats, 0.50), 1),
                "p95": round(_percentile(lats, 0.95), 1),
                "max": round(lats[-1], 1) if lats else 0,
            }

        # ── Alerts ──────────────────────────────────────────────────
        alerts: list[dict] = []

        # p95 search > 300ms
        for name, stats in latency.items():
            if "search" in name and stats["p95"] > 300:
                alerts.append({
                    "level": "danger",
                    "message": f"p95 latency for '{name}' is {stats['p95']}ms (>300ms)",
                })

        # Zero-result rate > 20%
        zero_rate = _safe_rate(n_zero, n_searches)
        if zero_rate > 20:
            alerts.append({
                "level": "warning",
                "message": f"Zero-result rate is {zero_rate}% (>20%)",
            })

        # Error rate > 5%
        error_rate = _safe_rate(n_errors, n_requests)
        if error_rate > 5:
            alerts.append({
                "level": "danger",
                "message": f"Error rate is {error_rate}% (>5%)",
            })

        # Tool error rate > 10%
        total_tool_calls = sum(tool_dist.values())
        tool_error_rate = _safe_rate(tool_errors, total_tool_calls)
        if tool_error_rate > 10:
            alerts.append({
                "level": "warning",
                "message": f"Tool error rate is {tool_error_rate}% (>10%)",
            })

        # ── Recent traces (drill-down) ──────────────────────────────
        trace_rows = c.execute("""
            SELECT t.trace_id, t.ts, t.mode, t.latency_ms, t.n_results,
                   t.status, t.query_raw, f.rating as feedback
            FROM traces t
            LEFT JOIN feedback f ON t.trace_id = f.trace_id
            WHERE t.ts > ? AND COALESCE(t.endpoint, '') NOT LIKE '/api/v1/stats/%'
            ORDER BY t.ts DESC
            LIMIT 50
        """, (day_ago,)).fetchall()

        recent_traces = [
            {
                "trace_id": r["trace_id"],
                "ts": r["ts"],
                "mode": r["mode"],
                "latency_ms": r["latency_ms"],
                "n_results": r["n_results"],
                "status": r["status"],
                "query_raw": r["query_raw"],
                "feedback": r["feedback"],
            }
            for r in trace_rows
        ]

    return {
        "period": "24h",
        "operations": {
            "requests": n_requests,
            "errors": n_errors,
            "error_rate": error_rate,
            "timeouts": n_timeouts,
            "fallbacks": n_fallbacks,
            "qps_hourly": qps_hourly,
        },
        "retrieval": {
            "zero_results": n_zero,
            "zero_rate": zero_rate,
            "source_contribution": source_contribution,
        },
        "answers": {
            "status": answer_status,
            "evidence_rate": evidence_rate,
            "feedback": feedback,
            "feedback_hourly": feedback_hourly,
        },
        "cost": {
            "total_tokens_in": cost_row["total_in"],
            "total_tokens_out": cost_row["total_out"],
            "avg_tokens_in": round(cost_row["avg_in"], 1),
            "avg_tokens_out": round(cost_row["avg_out"], 1),
        },
        "agent": {
            "total_runs": agent_total,
            "avg_steps": avg_steps,
            "tool_distribution": tool_dist,
            "tool_errors": tool_errors,
            "max_steps_hit": max_steps_hit,
        },
        "index": index_info,
        "latency": latency,
        "alerts": alerts,
        "recent_traces": recent_traces,
    }

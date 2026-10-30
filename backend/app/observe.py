"""Observability: span context manager + SQLite persistence.

Tables:
  traces  — one row per request/chat turn
  spans   — tree of timed operations within a trace

Usage:
    with span(trace_id, "search.dense", top_k=200) as attrs:
        results = do_search()
        attrs["n_results"] = len(results)
"""

import contextlib
import contextvars
import hashlib
import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path

from .config import settings

logger = logging.getLogger(__name__)

_current = contextvars.ContextVar("current_span_id", default=None)

_DB_PATH: str | None = None


def _db_path() -> str:
    global _DB_PATH
    if _DB_PATH is None:
        p = Path(settings.TRACES_DB)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent.parent / p
        p.parent.mkdir(parents=True, exist_ok=True)
        _DB_PATH = str(p)
    return _DB_PATH


def _init_db(path: str):
    """Create tables if they don't exist."""
    with sqlite3.connect(path, timeout=5) as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS traces (
                trace_id      TEXT PRIMARY KEY,
                ts            REAL,
                endpoint      TEXT,
                session_id    TEXT,
                workflow_id   TEXT,
                mode          TEXT,
                query_raw     TEXT,
                status        TEXT,
                latency_ms    REAL,
                tokens_in     INTEGER DEFAULT 0,
                tokens_out    INTEGER DEFAULT 0,
                index_version TEXT,
                encoder       TEXT,
                reranker      TEXT,
                prompt_version TEXT,
                llm_model     TEXT,
                vlm_model     TEXT,
                config_id     TEXT,
                temporal_strategy TEXT,
                n_results     INTEGER DEFAULT 0,
                feedback      INTEGER,
                feedback_note TEXT,
                error_label   TEXT
            );
            CREATE TABLE IF NOT EXISTS spans (
                span_id   TEXT PRIMARY KEY,
                trace_id  TEXT,
                parent_id TEXT,
                name      TEXT,
                start_ts  REAL,
                latency_ms REAL,
                attrs     TEXT,
                error     TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
            CREATE INDEX IF NOT EXISTS idx_traces_ts ON traces(ts);

            CREATE TABLE IF NOT EXISTS parsed_queries (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id   TEXT,
                session_id TEXT,
                source     TEXT,
                data       TEXT,
                ts         REAL
            );
            CREATE TABLE IF NOT EXISTS selections (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                kf_id      TEXT,
                action     TEXT,
                ts         REAL
            );
            CREATE TABLE IF NOT EXISTS answers (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id   TEXT,
                session_id TEXT,
                question   TEXT,
                answer     TEXT,
                status     TEXT,
                evidence   TEXT,
                mode       TEXT,
                ts         REAL
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id   TEXT,
                rating     INTEGER,
                note       TEXT,
                kf_ids     TEXT,
                client_total_ms REAL,
                ts         REAL
            );
        """)


_initialized = False


def _migrate_db(path: str):
    """Add columns introduced after initial schema."""
    new_cols = [
        ("traces", "workflow_id", "TEXT"),
        ("traces", "temporal_strategy", "TEXT"),
    ]
    with sqlite3.connect(path, timeout=5) as c:
        existing = {
            row[1]
            for row in c.execute("PRAGMA table_info(traces)").fetchall()
        }
        for table, col, dtype in new_cols:
            if col not in existing:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {dtype}")


def ensure_db():
    global _initialized
    if not _initialized:
        _init_db(_db_path())
        _migrate_db(_db_path())
        _initialized = True


@contextlib.contextmanager
def span(trace_id: str, name: str, **attrs):
    """Timed span that writes to SQLite on exit.

    Nests automatically via contextvars.
    Yields the attrs dict so the caller can add more data during the span.
    """
    ensure_db()
    sid = str(uuid.uuid4())
    parent = _current.get()
    token = _current.set(sid)
    start = time.time()
    t0 = time.perf_counter()
    err = None
    try:
        yield attrs
    except Exception as e:
        err = repr(e)
        raise
    finally:
        _current.reset(token)
        ms = (time.perf_counter() - t0) * 1000
        try:
            with sqlite3.connect(_db_path(), timeout=5) as c:
                c.execute(
                    "INSERT INTO spans VALUES (?,?,?,?,?,?,?,?)",
                    (sid, trace_id, parent, name, start, ms,
                     json.dumps(attrs, ensure_ascii=False, default=str), err),
                )
        except Exception as e:
            logger.warning("Failed to write span %s: %s", name, e)


def write_trace(trace_id: str, **kwargs):
    """Insert or update a trace row."""
    ensure_db()
    if "question" in kwargs and "query_raw" not in kwargs:
        kwargs["query_raw"] = kwargs["question"]
    fields = [
        "trace_id", "ts", "endpoint", "session_id", "workflow_id", "mode",
        "query_raw", "status", "latency_ms", "tokens_in", "tokens_out",
        "index_version", "encoder", "reranker", "prompt_version", "llm_model",
        "vlm_model", "config_id", "temporal_strategy", "n_results", "feedback",
        "feedback_note", "error_label",
    ]
    row = {"trace_id": trace_id, "ts": time.time()}
    row.update(kwargs)

    cols = [f for f in fields if f in row]
    placeholders = ",".join("?" for _ in cols)
    values = [row[f] for f in cols]
    col_str = ",".join(cols)
    updates = ", ".join(
        f"{col}=excluded.{col}" for col in cols if col not in ("trace_id", "ts")
    )
    conflict_action = f"DO UPDATE SET {updates}" if updates else "DO NOTHING"

    try:
        with sqlite3.connect(_db_path(), timeout=5) as c:
            c.execute(
                f"INSERT INTO traces ({col_str}) VALUES ({placeholders}) "
                f"ON CONFLICT(trace_id) {conflict_action}",
                values,
            )
    except Exception as e:
        logger.warning("Failed to write trace %s: %s", trace_id, e)


def new_trace_id() -> str:
    return str(uuid.uuid4())


def config_id(flags: dict) -> str:
    """Deterministic 8-char hash of a flags dict."""
    raw = json.dumps(flags, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode()).hexdigest()[:8]


def write_answer(
    trace_id: str,
    session_id: str = "",
    question: str = "",
    answer: str = "",
    status: str = "",
    evidence: list[str] | None = None,
    mode: str = "",
):
    """Record an answer to the answers table."""
    ensure_db()
    try:
        with sqlite3.connect(_db_path(), timeout=5) as c:
            c.execute(
                "INSERT INTO answers (trace_id, session_id, question, answer, status, evidence, mode, ts) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (trace_id, session_id, question, answer, status,
                 json.dumps(evidence or []), mode, time.time()),
            )
    except Exception as e:
        logger.warning("Failed to write answer for trace %s: %s", trace_id, e)


def write_feedback(
    trace_id: str,
    rating: int = 0,
    note: str = "",
    kf_ids: list[str] | None = None,
    client_total_ms: float = 0,
):
    """Record user feedback."""
    ensure_db()
    try:
        with sqlite3.connect(_db_path(), timeout=5) as c:
            c.execute(
                "INSERT INTO feedback (trace_id, rating, note, kf_ids, client_total_ms, ts) "
                "VALUES (?,?,?,?,?,?)",
                (trace_id, rating, note,
                 json.dumps(kf_ids or []), client_total_ms, time.time()),
            )
    except Exception as e:
        logger.warning("Failed to write feedback for trace %s: %s", trace_id, e)


def get_traces(limit: int = 50, offset: int = 0) -> list[dict]:
    """Get recent traces for the trace viewer."""
    ensure_db()
    with sqlite3.connect(_db_path(), timeout=5) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT * FROM traces ORDER BY ts DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


def get_trace(trace_id: str) -> dict | None:
    """Get a single trace with its spans."""
    ensure_db()
    with sqlite3.connect(_db_path(), timeout=5) as c:
        c.row_factory = sqlite3.Row
        trace_row = c.execute(
            "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        span_rows = c.execute(
            "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_ts",
            (trace_id,),
        ).fetchall()
    if not trace_row:
        return None
    return {
        "trace": dict(trace_row),
        "spans": [dict(r) for r in span_rows],
    }


def get_answers(session_id: str | None = None, limit: int = 100) -> list[dict]:
    """Get answers, optionally filtered by session."""
    ensure_db()
    with sqlite3.connect(_db_path(), timeout=5) as c:
        c.row_factory = sqlite3.Row
        if session_id:
            rows = c.execute(
                "SELECT * FROM answers WHERE session_id = ? ORDER BY ts DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM answers ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


def get_workflow_traces(workflow_id: str) -> list[dict]:
    """Get all traces linked by a manual-mode workflow_id."""
    ensure_db()
    with sqlite3.connect(_db_path(), timeout=5) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT * FROM traces WHERE workflow_id = ? ORDER BY ts",
            (workflow_id,),
        ).fetchall()
    return [dict(r) for r in rows]

"""Generate experiment reports from trace data.

Reads traces from PostgreSQL, computes metrics over date/config ranges,
outputs structured JSON + Markdown + CSV.

Usage:
    python -m backend.scripts.generate_report --config-id <id> --since 2025-01-01
    python -m backend.scripts.generate_report --experiment R3 --compare-to R1
    python -m backend.scripts.generate_report --mode manual --mode agentic --compare
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median, quantiles


def get_db_path() -> str:
    return str(Path(__file__).resolve().parent.parent / "data" / "traces.db")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def fetch_traces(
    conn: sqlite3.Connection,
    config_id: str | None = None,
    mode: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> list[dict]:
    """Fetch traces matching filters."""
    query = "SELECT * FROM traces WHERE 1=1"
    params: list = []

    if config_id:
        query += " AND config_id = ?"
        params.append(config_id)
    if mode:
        query += " AND mode = ?"
        params.append(mode)
    if since:
        query += " AND ts >= ?"
        params.append(datetime.fromisoformat(since).timestamp())
    if until:
        query += " AND ts <= ?"
        params.append(datetime.fromisoformat(until).timestamp())

    query += " ORDER BY ts DESC"

    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def fetch_spans(conn: sqlite3.Connection, trace_ids: list[str]) -> list[dict]:
    """Fetch spans for a list of trace IDs."""
    if not trace_ids:
        return []
    placeholders = ",".join(["?"] * len(trace_ids))
    rows = conn.execute(
        f"SELECT * FROM spans WHERE trace_id IN ({placeholders}) ORDER BY ts",
        trace_ids,
    ).fetchall()
    return [dict(r) for r in rows]


def compute_latency_stats(spans: list[dict]) -> dict[str, dict]:
    """Compute p50/p95/p99/max latency per span name."""
    by_name: dict[str, list[float]] = {}
    for s in spans:
        name = s.get("name", "unknown")
        ms = s.get("latency_ms", 0.0)
        if ms is not None and ms > 0:
            by_name.setdefault(name, []).append(ms)

    stats = {}
    for name, values in sorted(by_name.items()):
        values.sort()
        n = len(values)
        q = quantiles(values, n=100) if n >= 2 else values
        stats[name] = {
            "count": n,
            "p50": q[49] if len(q) > 49 else median(values),
            "p95": q[94] if len(q) > 94 else max(values),
            "p99": q[98] if len(q) > 98 else max(values),
            "max": max(values),
            "mean": round(mean(values), 2),
        }
    return stats


def compute_answer_stats(traces: list[dict]) -> dict:
    """Compute answer status distribution."""
    status_counts: dict[str, int] = {}
    for t in traces:
        status = t.get("status", "unknown")
        if status:
            status_counts[status] = status_counts.get(status, 0) + 1
    return status_counts


def compute_overview(traces: list[dict]) -> dict:
    """Compute overview statistics."""
    total = len(traces)
    errors = sum(1 for t in traces if t.get("status") == "error")
    zero_results = sum(1 for t in traces if t.get("n_results", -1) == 0)

    latencies = [t.get("latency_ms", 0) for t in traces if t.get("latency_ms")]

    return {
        "total_requests": total,
        "errors": errors,
        "error_rate": round(errors / total, 4) if total else 0,
        "zero_result_rate": round(zero_results / total, 4) if total else 0,
        "latency_p50": round(median(latencies), 2) if latencies else 0,
        "latency_p95": round(
            quantiles(sorted(latencies), n=100)[94] if len(latencies) >= 2 else max(latencies, default=0), 2
        ),
    }


def generate_report(
    traces: list[dict],
    spans: list[dict],
    experiment_id: str = "",
    config_diff: str = "",
    dataset: str = "",
) -> dict:
    """Generate a full report dict."""
    overview = compute_overview(traces)
    latency = compute_latency_stats(spans)
    answer = compute_answer_stats(traces)

    return {
        "experiment": experiment_id or "unnamed",
        "date": datetime.now().isoformat()[:10],
        "config_diff": config_diff,
        "dataset": dataset,
        "n_traces": len(traces),
        "overview": overview,
        "latency_by_span": latency,
        "answer_status": answer,
    }


def report_to_markdown(report: dict) -> str:
    """Convert report dict to Markdown."""
    lines = [
        f"# Experiment Report: {report['experiment']}",
        f"",
        f"**Date**: {report['date']}",
        f"**Config diff**: {report.get('config_diff', 'N/A')}",
        f"**Dataset**: {report.get('dataset', 'N/A')}",
        f"**Traces**: {report['n_traces']}",
        f"",
        f"## Overview",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]
    for k, v in report["overview"].items():
        lines.append(f"| {k} | {v} |")

    lines.extend([
        f"",
        f"## Latency by Span (ms)",
        f"",
        f"| Span | Count | p50 | p95 | p99 | Max |",
        f"|------|-------|-----|-----|-----|-----|",
    ])
    for name, s in report["latency_by_span"].items():
        lines.append(
            f"| {name} | {s['count']} | {s['p50']:.1f} | {s['p95']:.1f} | {s['p99']:.1f} | {s['max']:.1f} |"
        )

    if report["answer_status"]:
        lines.extend([
            f"",
            f"## Answer Status",
            f"",
            f"| Status | Count |",
            f"|--------|-------|",
        ])
        for status, count in report["answer_status"].items():
            lines.append(f"| {status} | {count} |")

    return "\n".join(lines) + "\n"


def report_to_csv(report: dict, output_dir: Path) -> list[Path]:
    """Write report data as CSV files. Returns list of created files."""
    files = []

    # Overview CSV
    overview_path = output_dir / f"{report['experiment']}_overview.csv"
    with open(overview_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in report["overview"].items():
            w.writerow([k, v])
    files.append(overview_path)

    # Latency CSV
    latency_path = output_dir / f"{report['experiment']}_latency.csv"
    with open(latency_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["span", "count", "p50", "p95", "p99", "max", "mean"])
        for name, s in report["latency_by_span"].items():
            w.writerow([name, s["count"], s["p50"], s["p95"], s["p99"], s["max"], s["mean"]])
    files.append(latency_path)

    return files


def compare_reports(report_a: dict, report_b: dict) -> str:
    """Generate a comparison Markdown between two reports."""
    lines = [
        f"# Comparison: {report_a['experiment']} vs {report_b['experiment']}",
        f"",
        f"**Date**: {datetime.now().isoformat()[:10]}",
        f"",
        f"## Overview",
        f"",
        f"| Metric | {report_a['experiment']} | {report_b['experiment']} | Delta |",
        f"|--------|{'------|' * 3}",
    ]

    all_keys = set(report_a["overview"].keys()) | set(report_b["overview"].keys())
    for k in sorted(all_keys):
        va = report_a["overview"].get(k, 0)
        vb = report_b["overview"].get(k, 0)
        delta = vb - va if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else "N/A"
        if isinstance(delta, float):
            delta = f"{delta:+.4f}"
        elif isinstance(delta, int):
            delta = f"{delta:+d}"
        lines.append(f"| {k} | {va} | {vb} | {delta} |")

    lines.extend([
        f"",
        f"## Latency Comparison (p95 ms)",
        f"",
        f"| Span | {report_a['experiment']} | {report_b['experiment']} | Delta |",
        f"|------|{'------|' * 3}",
    ])

    all_spans = set(report_a["latency_by_span"].keys()) | set(report_b["latency_by_span"].keys())
    for name in sorted(all_spans):
        sa = report_a["latency_by_span"].get(name, {})
        sb = report_b["latency_by_span"].get(name, {})
        pa = sa.get("p95", 0)
        pb = sb.get("p95", 0)
        delta = pb - pa
        lines.append(f"| {name} | {pa:.1f} | {pb:.1f} | {delta:+.1f} |")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate experiment report from traces")
    parser.add_argument("--config-id", help="Filter by config_id")
    parser.add_argument("--mode", action="append", help="Filter by mode (can specify multiple)")
    parser.add_argument("--since", help="Start date (ISO format)")
    parser.add_argument("--until", help="End date (ISO format)")
    parser.add_argument("--experiment", default="unnamed", help="Experiment ID for the report")
    parser.add_argument("--config-diff", default="", help="Description of config change")
    parser.add_argument("--dataset", default="", help="Dataset name")
    parser.add_argument("--output-dir", default="reports", help="Output directory")
    parser.add_argument("--compare", action="store_true", help="Compare two modes side by side")
    parser.add_argument("--format", choices=["json", "markdown", "csv", "all"], default="all")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = connect()

    if args.compare and args.mode and len(args.mode) >= 2:
        reports = []
        for mode in args.mode[:2]:
            traces = fetch_traces(conn, config_id=args.config_id, mode=mode,
                                  since=args.since, until=args.until)
            trace_ids = [t["trace_id"] for t in traces]
            spans = fetch_spans(conn, trace_ids)
            report = generate_report(traces, spans,
                                     experiment_id=f"{args.experiment}_{mode}",
                                     config_diff=args.config_diff,
                                     dataset=args.dataset)
            reports.append(report)

        comparison = compare_reports(reports[0], reports[1])
        comp_path = output_dir / f"{args.experiment}_comparison.md"
        comp_path.write_text(comparison, encoding="utf-8")
        print(f"Comparison: {comp_path}")

        for r in reports:
            _write_report(r, output_dir, args.format)
    else:
        mode = args.mode[0] if args.mode else None
        traces = fetch_traces(conn, config_id=args.config_id, mode=mode,
                              since=args.since, until=args.until)

        if not traces:
            print("No traces found matching filters.", file=sys.stderr)
            sys.exit(1)

        trace_ids = [t["trace_id"] for t in traces]
        spans = fetch_spans(conn, trace_ids)

        report = generate_report(traces, spans,
                                 experiment_id=args.experiment,
                                 config_diff=args.config_diff,
                                 dataset=args.dataset)
        _write_report(report, output_dir, args.format)

    conn.close()


def _write_report(report: dict, output_dir: Path, fmt: str):
    """Write report in the requested format(s)."""
    name = report["experiment"]

    if fmt in ("json", "all"):
        p = output_dir / f"{name}.json"
        p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON: {p}")

    if fmt in ("markdown", "all"):
        p = output_dir / f"{name}.md"
        p.write_text(report_to_markdown(report), encoding="utf-8")
        print(f"Markdown: {p}")

    if fmt in ("csv", "all"):
        files = report_to_csv(report, output_dir)
        for f in files:
            print(f"CSV: {f}")


if __name__ == "__main__":
    main()

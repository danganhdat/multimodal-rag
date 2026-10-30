import { useEffect, useState, useCallback, useMemo } from "react";

/* ─── Types ──────────────────────────────────────────────────────── */

interface Alert {
  level: "danger" | "warning";
  message: string;
}

interface LatencyEntry {
  count: number;
  p50: number;
  p95: number;
  max: number;
}

interface TraceRow {
  trace_id: string;
  ts: number;
  mode: string;
  latency_ms: number;
  n_results: number;
  status: string;
  query_raw: string;
  feedback: number | null;
}

interface HourlyQPS {
  hour: string;
  count: number;
}

interface FeedbackHourly {
  hour: string;
  up: number;
  down: number;
}

interface DashboardData {
  period: string;
  operations: {
    requests: number;
    errors: number;
    error_rate: number;
    timeouts: number;
    fallbacks: number;
    qps_hourly: HourlyQPS[];
  };
  retrieval: {
    zero_results: number;
    zero_rate: number;
    source_contribution: Record<string, number>;
  };
  answers: {
    status: Record<string, number>;
    evidence_rate: number;
    feedback: Record<string, number>;
    feedback_hourly: FeedbackHourly[];
  };
  cost: {
    total_tokens_in: number;
    total_tokens_out: number;
    avg_tokens_in: number;
    avg_tokens_out: number;
  };
  agent: {
    total_runs: number;
    avg_steps: number;
    tool_distribution: Record<string, number>;
    tool_errors: number;
    max_steps_hit: number;
  };
  index: Record<string, string>;
  latency: Record<string, LatencyEntry>;
  alerts: Alert[];
  recent_traces: TraceRow[];
}

/* ─── Trace filter type ──────────────────────────────────────────── */

type TraceFilter = "all" | "errors" | "zero" | "thumbs_down" | "high_latency";

/* ─── Helpers ────────────────────────────────────────────────────── */

function fmtTime(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function fmtNumber(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return String(n);
}

function copyText(text: string) {
  navigator.clipboard.writeText(text).catch(() => {});
}

/* ─── Section Header ─────────────────────────────────────────────── */

function SectionHeader({ title }: { title: string }) {
  return <h3 className="title is-6 mb-3 mt-5">{title}</h3>;
}

/* ─── Stat Card ──────────────────────────────────────────────────── */

function StatCard({
  label,
  value,
  suffix,
  color,
}: {
  label: string;
  value: string | number;
  suffix?: string;
  color?: string;
}) {
  return (
    <div className="column is-3">
      <div
        className={`notification ${color || "is-light"} py-3 px-3 has-text-centered`}
        style={{ borderRadius: 8 }}
      >
        <p className="is-size-3 has-text-weight-bold" style={{ lineHeight: 1.2 }}>
          {value}
          {suffix && <span className="is-size-6 has-text-weight-normal"> {suffix}</span>}
        </p>
        <p className="is-size-7 mt-1">{label}</p>
      </div>
    </div>
  );
}

/* ─── QPS Bar Chart ──────────────────────────────────────────────── */

function QPSChart({ data }: { data: HourlyQPS[] }) {
  const maxCount = Math.max(...data.map((d) => d.count), 1);
  const BAR_MAX_H = 80;
  return (
    <div className="box">
      <p className="is-size-7 has-text-weight-semibold mb-2">Requests per Hour (last 24h)</p>
      <div
        style={{
          display: "flex",
          alignItems: "flex-end",
          gap: 2,
          height: BAR_MAX_H + 20,
          overflowX: "auto",
        }}
      >
        {data.map((d) => {
          const h = Math.max((d.count / maxCount) * BAR_MAX_H, 2);
          const hourLabel = d.hour.slice(11, 16); // "HH:MM"
          return (
            <div
              key={d.hour}
              style={{ display: "flex", flexDirection: "column", alignItems: "center", flex: 1, minWidth: 18 }}
              title={`${d.hour}: ${d.count} requests`}
            >
              <span className="is-size-7" style={{ fontSize: 9, marginBottom: 2 }}>
                {d.count > 0 ? d.count : ""}
              </span>
              <div
                style={{
                  width: "100%",
                  height: h,
                  background: "hsl(217, 71%, 53%)",
                  borderRadius: "3px 3px 0 0",
                  minWidth: 8,
                }}
              />
              <span style={{ fontSize: 8, marginTop: 2, color: "#888" }}>{hourLabel}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ─── Latency Table ──────────────────────────────────────────────── */

function LatencyTable({ data }: { data: Record<string, LatencyEntry> }) {
  const entries = Object.entries(data).sort((a, b) => b[1].p95 - a[1].p95);
  const maxP50 = Math.max(...entries.map(([, s]) => s.p50), 1);

  return (
    <div className="table-container">
      <table className="table is-narrow is-striped is-fullwidth is-hoverable">
        <thead>
          <tr>
            <th>Span</th>
            <th>Count</th>
            <th style={{ minWidth: 160 }}>p50 (ms)</th>
            <th>p95 (ms)</th>
            <th>Max (ms)</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([name, s]) => {
            const barWidth = Math.max((s.p50 / maxP50) * 100, 2);
            const p95Danger = s.p95 > 300;
            return (
              <tr key={name}>
                <td className="is-size-7">{name}</td>
                <td className="is-size-7">{s.count}</td>
                <td className="is-size-7">
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <div
                      style={{
                        height: 10,
                        width: `${barWidth}%`,
                        background: "hsl(217, 71%, 53%)",
                        borderRadius: 3,
                        minWidth: 4,
                      }}
                    />
                    <span>{s.p50.toFixed(1)}</span>
                  </div>
                </td>
                <td
                  className="is-size-7"
                  style={{
                    color: p95Danger ? "hsl(348, 100%, 61%)" : undefined,
                    fontWeight: p95Danger ? "bold" : undefined,
                  }}
                >
                  {s.p95.toFixed(1)}
                </td>
                <td className="is-size-7">{s.max.toFixed(1)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ─── Drill-down Trace Table ─────────────────────────────────────── */

function TraceTable({
  traces,
  filter,
  setFilter,
}: {
  traces: TraceRow[];
  filter: TraceFilter;
  setFilter: (f: TraceFilter) => void;
}) {
  const filtered = useMemo(() => {
    switch (filter) {
      case "errors":
        return traces.filter((t) => t.status === "error" || Number(t.status) >= 400);
      case "zero":
        return traces.filter((t) => ["search", "search_parsed", "image_search"].includes(t.mode) && t.n_results === 0);
      case "thumbs_down":
        return traces.filter((t) => t.feedback === -1);
      case "high_latency":
        return traces.filter((t) => t.latency_ms > 1000);
      default:
        return traces;
    }
  }, [traces, filter]);

  const filters: { key: TraceFilter; label: string }[] = [
    { key: "all", label: "All" },
    { key: "errors", label: "Errors" },
    { key: "zero", label: "Zero Results" },
    { key: "thumbs_down", label: "Thumbs Down" },
    { key: "high_latency", label: "High Latency" },
  ];

  const feedbackIcon = (fb: number | null) => {
    if (fb === 1) return <span style={{ color: "hsl(141, 53%, 53%)" }}>+1</span>;
    if (fb === -1) return <span style={{ color: "hsl(348, 100%, 61%)" }}>-1</span>;
    return <span style={{ color: "#ccc" }}>--</span>;
  };

  return (
    <div>
      <div className="buttons are-small mb-3">
        {filters.map((f) => (
          <button
            key={f.key}
            className={`button is-small ${filter === f.key ? "is-link" : "is-light"}`}
            onClick={() => setFilter(f.key)}
          >
            {f.label}
          </button>
        ))}
      </div>
      <div className="table-container">
        <table className="table is-narrow is-striped is-fullwidth is-hoverable">
          <thead>
            <tr>
              <th>Time</th>
              <th>Mode</th>
              <th>Latency</th>
              <th>Results</th>
              <th>Status</th>
              <th>Query</th>
              <th>FB</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 && (
              <tr>
                <td colSpan={7} className="has-text-centered has-text-grey-light is-size-7">
                  No traces match this filter
                </td>
              </tr>
            )}
            {filtered.map((t) => (
              <tr
                key={t.trace_id}
                style={{ cursor: "pointer" }}
                title={`Click to copy trace_id: ${t.trace_id}`}
                onClick={() => copyText(t.trace_id)}
              >
                <td className="is-size-7">{fmtTime(t.ts)}</td>
                <td className="is-size-7">
                  <span className="tag is-small is-light">{t.mode || "--"}</span>
                </td>
                <td
                  className="is-size-7"
                  style={{
                    color: t.latency_ms > 1000 ? "hsl(348, 100%, 61%)" : undefined,
                    fontWeight: t.latency_ms > 1000 ? "bold" : undefined,
                  }}
                >
                  {t.latency_ms != null ? t.latency_ms.toFixed(0) + "ms" : "--"}
                </td>
                <td className="is-size-7">{t.n_results ?? "--"}</td>
                <td className="is-size-7">
                  <span
                    className={`tag is-small ${
                      t.status === "error" || Number(t.status) >= 400
                        ? "is-danger"
                        : t.status === "ok"
                          ? "is-success"
                          : "is-light"
                    }`}
                  >
                    {t.status || "--"}
                  </span>
                </td>
                <td className="is-size-7" style={{ maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {t.query_raw ? t.query_raw.slice(0, 50) : "--"}
                </td>
                <td className="is-size-7">{feedbackIcon(t.feedback)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ─── Main Dashboard ─────────────────────────────────────────────── */

export default function DashboardPage() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [traceFilter, setTraceFilter] = useState<TraceFilter>("all");
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  const fetchData = useCallback(() => {
    fetch("/api/v1/stats/dashboard")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d) => {
        setData(d);
        setError(null);
        setLastRefresh(new Date());
      })
      .catch((e) => setError(e.message));
  }, []);

  // Initial fetch + 30s auto-refresh
  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30_000);
    return () => clearInterval(interval);
  }, [fetchData]);

  if (error && !data) {
    return (
      <div className="p-4">
        <article className="message is-danger">
          <div className="message-body">Failed to load dashboard: {error}</div>
        </article>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="p-4 has-text-centered">
        <progress className="progress is-small is-primary" max="100" />
      </div>
    );
  }

  const ops = data.operations ?? { requests: 0, errors: 0, error_rate: 0, timeouts: 0, fallbacks: 0, qps_hourly: [] as HourlyQPS[] };
  const ret = data.retrieval ?? { zero_results: 0, zero_rate: 0, source_contribution: {} };
  const ans = data.answers ?? { status: {}, evidence_rate: 0, feedback: {}, feedback_hourly: [] as FeedbackHourly[] };
  const cost = data.cost ?? { total_tokens_in: 0, total_tokens_out: 0, avg_tokens_in: 0, avg_tokens_out: 0 };
  const agent = data.agent ?? { total_runs: 0, avg_steps: 0, tool_distribution: {}, tool_errors: 0, max_steps_hit: 0 };

  return (
    <div className="p-4" style={{ maxWidth: 1100, margin: "0 auto" }}>
      {/* Header */}
      <div className="level mb-4">
        <div className="level-left">
          <h2 className="title is-5 mb-0">Dashboard ({data.period})</h2>
        </div>
        <div className="level-right">
          {lastRefresh && (
            <span className="is-size-7 has-text-grey">
              Last refresh: {lastRefresh.toLocaleTimeString()} (auto 30s)
            </span>
          )}
        </div>
      </div>

      {/* 1. Alerts */}
      {(data.alerts ?? []).length > 0 && (
        <div className="mb-4">
          {(data.alerts ?? []).map((a, i) => (
            <article key={i} className={`message is-${a.level} mb-2`}>
              <div className="message-body py-2 px-3 is-size-7">
                <strong>{a.level === "danger" ? "ALERT" : "WARNING"}:</strong> {a.message}
              </div>
            </article>
          ))}
        </div>
      )}

      {/* 2. Operations */}
      <SectionHeader title="Operations" />
      <div className="columns is-multiline">
        <StatCard label="Requests" value={fmtNumber(ops.requests)} />
        <StatCard
          label="Error Rate"
          value={ops.error_rate.toFixed(1)}
          suffix="%"
          color={ops.error_rate > 5 ? "is-danger" : ops.errors > 0 ? "is-warning" : "is-light"}
        />
        <StatCard label="Timeouts" value={ops.timeouts} color={ops.timeouts > 0 ? "is-warning" : "is-light"} />
        <StatCard label="Fallbacks" value={ops.fallbacks} color={ops.fallbacks > 0 ? "is-info" : "is-light"} />
      </div>
      {ops.qps_hourly.length > 0 && <QPSChart data={ops.qps_hourly} />}

      {/* 3. Latency */}
      <SectionHeader title="Latency by Span" />
      {Object.keys(data.latency).length > 0 ? (
        <div className="box">
          <LatencyTable data={data.latency} />
        </div>
      ) : (
        <p className="is-size-7 has-text-grey">No latency data</p>
      )}

      {/* 4. Retrieval Quality */}
      <SectionHeader title="Retrieval Quality" />
      <div className="columns">
        <div className="column is-4">
          <div
            className={`notification ${ret.zero_rate > 20 ? "is-warning" : "is-light"} has-text-centered py-3`}
            style={{ borderRadius: 8 }}
          >
            <p className="is-size-3 has-text-weight-bold">{ret.zero_rate.toFixed(1)}%</p>
            <p className="is-size-7">Zero-Result Rate ({ret.zero_results} queries)</p>
          </div>
        </div>
        {Object.keys(ret.source_contribution).length > 0 && (
          <div className="column is-8">
            <div className="box">
              <p className="is-size-7 has-text-weight-semibold mb-2">Source Contribution</p>
              <div className="tags">
                {Object.entries(ret.source_contribution).map(([src, cnt]) => (
                  <span key={src} className="tag is-info is-light">
                    {src}: {cnt}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 5. Answer Quality */}
      <SectionHeader title="Answer Quality" />
      <div className="columns is-multiline">
        <div className="column is-6">
          <div className="box">
            <p className="is-size-7 has-text-weight-semibold mb-2">Status Distribution</p>
            <div className="tags">
              {Object.entries(ans.status).map(([status, count]) => (
                <span
                  key={status}
                  className={`tag ${
                    status === "answered"
                      ? "is-success"
                      : status === "uncertain"
                        ? "is-warning"
                        : status === "not_found"
                          ? "is-danger"
                          : "is-light"
                  }`}
                >
                  {status}: {count}
                </span>
              ))}
              {Object.keys(ans.status).length === 0 && (
                <span className="is-size-7 has-text-grey">No answers yet</span>
              )}
            </div>
          </div>
        </div>
        <div className="column is-3">
          <div className="notification is-light has-text-centered py-3" style={{ borderRadius: 8 }}>
            <p className="is-size-3 has-text-weight-bold">{ans.evidence_rate.toFixed(0)}%</p>
            <p className="is-size-7">Evidence Rate</p>
          </div>
        </div>
        <div className="column is-3">
          <div className="box has-text-centered">
            <p className="is-size-7 has-text-weight-semibold mb-2">Feedback</p>
            <span className="tag is-success is-light mr-1">+1: {ans.feedback["1"] || 0}</span>
            <span className="tag is-danger is-light">-1: {ans.feedback["-1"] || 0}</span>
          </div>
        </div>
      </div>

      {/* 6. Cost */}
      <SectionHeader title="Cost (Tokens)" />
      <div className="columns is-multiline">
        <StatCard label="Total Tokens In" value={fmtNumber(cost.total_tokens_in)} />
        <StatCard label="Total Tokens Out" value={fmtNumber(cost.total_tokens_out)} />
        <StatCard label="Avg Tokens In" value={cost.avg_tokens_in.toFixed(0)} />
        <StatCard label="Avg Tokens Out" value={cost.avg_tokens_out.toFixed(0)} />
      </div>

      {/* 7. Agent */}
      <SectionHeader title="Agent" />
      <div className="columns is-multiline">
        <StatCard label="Agent Runs" value={agent.total_runs} />
        <StatCard label="Avg Steps" value={agent.avg_steps.toFixed(1)} />
        <StatCard
          label="Tool Errors"
          value={agent.tool_errors}
          color={agent.tool_errors > 0 ? "is-warning" : "is-light"}
        />
        <StatCard
          label="Max Steps Hit"
          value={agent.max_steps_hit}
          color={agent.max_steps_hit > 0 ? "is-warning" : "is-light"}
        />
      </div>
      {Object.keys(agent.tool_distribution).length > 0 && (
        <div className="box">
          <p className="is-size-7 has-text-weight-semibold mb-2">Tool Distribution</p>
          <div className="tags">
            {Object.entries(agent.tool_distribution)
              .sort((a, b) => b[1] - a[1])
              .map(([tool, cnt]) => (
                <span key={tool} className="tag is-link is-light">
                  {tool}: {cnt}
                </span>
              ))}
          </div>
        </div>
      )}

      {/* 8. Data & Index */}
      <SectionHeader title="Data & Index" />
      {Object.keys(data.index).length > 0 ? (
        <div className="box">
          <div className="tags">
            {Object.entries(data.index).map(([k, v]) =>
              v ? (
                <span key={k} className="tag is-light">
                  {k}: {v}
                </span>
              ) : null,
            )}
          </div>
        </div>
      ) : (
        <p className="is-size-7 has-text-grey">No index info</p>
      )}

      {/* 9. Drill-down */}
      <SectionHeader title="Recent Traces" />
      <div className="box">
        <TraceTable traces={data.recent_traces ?? []} filter={traceFilter} setFilter={setTraceFilter} />
      </div>

      {/* Footer spacing */}
      <div style={{ height: 40 }} />
    </div>
  );
}

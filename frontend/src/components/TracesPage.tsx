import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";

/* ── types ────────────────────────────────────────────────── */

interface Trace {
  trace_id: string;
  ts: number;
  endpoint: string;
  mode: string;
  latency_ms: number;
  n_results: number;
  status: string;
  query_raw: string;
}

interface SpanRow {
  span_id: string;
  trace_id: string;
  parent_id: string | null;
  name: string;
  latency_ms: number;
  attrs: string;
  error: string | null;
}

type FilterKey = "all" | "errors" | "zero" | "slow";

/* ── helpers ──────────────────────────────────────────────── */

function spanBorderColor(name: string, error: string | null): string {
  if (error) return "hsl(348, 100%, 61%)";
  if (/^(search|retrieve)\./.test(name)) return "hsl(204, 86%, 53%)";
  if (/^fusion\./.test(name)) return "hsl(141, 53%, 53%)";
  if (/^rerank\./.test(name)) return "hsl(48, 100%, 29%)";
  if (/^(agent|llm|vlm|tool)\./.test(name)) return "hsl(271, 76%, 53%)";
  return "hsl(0, 0%, 71%)";
}

function formatAttrValue(v: unknown): string {
  if (v === null || v === undefined) return "null";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(2);
  if (typeof v === "boolean") return String(v);
  if (Array.isArray(v)) return v.map((x) => String(x)).join(", ");
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max) + "…" : s;
}

/* ── MermaidDiagram ───────────────────────────────────────── */

function MermaidDiagram({ code }: { code: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const reactId = useId().replace(/:/g, "_");
  const [renderFailed, setRenderFailed] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!code || !containerRef.current) return;
    let cancelled = false;
    setRenderFailed(false);
    setLoading(true);

    const timeout = setTimeout(() => {
      if (!cancelled) { setRenderFailed(true); setLoading(false); }
    }, 8000);

    (async () => {
      try {
        const mermaidUrl = "https://cdnjs.cloudflare.com/ajax/libs/mermaid/11.4.1/mermaid.esm.min.mjs";
        const mermaid = await import(/* @vite-ignore */ mermaidUrl);
        if (cancelled) return;
        mermaid.default.initialize({ startOnLoad: false, theme: "default" });
        const { svg } = await mermaid.default.render(
          "mmd" + reactId + Date.now(),
          code,
        );
        if (!cancelled && containerRef.current) {
          containerRef.current.innerHTML = svg;
          setLoading(false);
        }
      } catch {
        if (!cancelled) { setRenderFailed(true); setLoading(false); }
      } finally {
        clearTimeout(timeout);
      }
    })();

    return () => { cancelled = true; clearTimeout(timeout); };
  }, [code, reactId]);

  if (renderFailed) {
    return (
      <div style={{ background: "#f5f5f5", borderRadius: "6px", padding: "0.75rem" }}>
        <p className="is-size-7 has-text-grey mb-2">
          Mermaid diagram could not render (CDN unreachable). Raw diagram code:
        </p>
        <pre style={{ whiteSpace: "pre-wrap", fontSize: "0.75rem", maxHeight: "50vh", overflow: "auto", background: "#fafafa", padding: "0.5rem", borderRadius: "4px" }}>
          {code}
        </pre>
      </div>
    );
  }

  return (
    <>
      {loading && <progress className="progress is-small is-primary" max="100" />}
      <div
        ref={containerRef}
        style={{
          overflow: "auto",
          maxHeight: "50vh",
          background: "#fafafa",
          borderRadius: "6px",
          padding: "0.75rem",
          display: loading ? "none" : "block",
        }}
      />
    </>
  );
}

/* ── SpanCard ─────────────────────────────────────────────── */

function SpanCard({
  span,
  maxLatency,
}: {
  span: SpanRow;
  maxLatency: number;
}) {
  const [open, setOpen] = useState(false);

  let parsedAttrs: Record<string, unknown> = {};
  try {
    parsedAttrs = JSON.parse(span.attrs || "{}");
  } catch {
    /* ignore */
  }
  const attrKeys = Object.keys(parsedAttrs);
  const barWidth = maxLatency > 0 ? (span.latency_ms / maxLatency) * 100 : 0;
  const borderColor = spanBorderColor(span.name, span.error);

  return (
    <div
      className="box py-2 px-3 mb-2"
      style={{
        borderLeft: `4px solid ${borderColor}`,
        marginLeft: span.parent_id ? "1.5rem" : 0,
      }}
    >
      {/* header row */}
      <div className="is-flex is-justify-content-space-between is-align-items-center">
        <span className="is-size-7 has-text-weight-semibold">{span.name}</span>
        <span className="tag is-light is-small">
          {span.latency_ms?.toFixed(1)} ms
        </span>
      </div>

      {/* latency bar */}
      <div
        style={{
          height: "3px",
          background: "#eee",
          borderRadius: "2px",
          marginTop: "4px",
          marginBottom: "4px",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${barWidth}%`,
            background: borderColor,
            borderRadius: "2px",
            transition: "width 0.3s",
          }}
        />
      </div>

      {/* error */}
      {span.error && (
        <p className="is-size-7 has-text-danger mb-1">{span.error}</p>
      )}

      {/* collapsible attributes */}
      {attrKeys.length > 0 && (
        <>
          <button
            className="button is-ghost is-small px-0 py-0"
            style={{ height: "auto", textDecoration: "none" }}
            onClick={() => setOpen((o) => !o)}
          >
            <span className="icon is-small" style={{ fontSize: "0.65rem" }}>
              {open ? "▼" : "▶"}
            </span>
            <span className="is-size-7 has-text-grey">
              {attrKeys.length} attribute{attrKeys.length > 1 ? "s" : ""}
            </span>
          </button>

          {open && (
            <table
              className="table is-narrow is-fullwidth mt-1 mb-0"
              style={{ fontSize: "0.75rem" }}
            >
              <tbody>
                {attrKeys.map((k) => (
                  <tr key={k}>
                    <td
                      className="has-text-weight-medium"
                      style={{
                        whiteSpace: "nowrap",
                        paddingRight: "0.75rem",
                        border: "none",
                        paddingTop: "2px",
                        paddingBottom: "2px",
                      }}
                    >
                      {k}
                    </td>
                    <td
                      className="has-text-grey-dark"
                      style={{
                        wordBreak: "break-all",
                        border: "none",
                        paddingTop: "2px",
                        paddingBottom: "2px",
                      }}
                    >
                      {formatAttrValue(parsedAttrs[k])}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}
    </div>
  );
}

/* ── TracesPage ───────────────────────────────────────────── */

export default function TracesPage() {
  const [traces, setTraces] = useState<Trace[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [spans, setSpans] = useState<SpanRow[]>([]);
  const [mermaidCode, setMermaidCode] = useState("");
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState<FilterKey>("all");

  /* fetch trace list */
  useEffect(() => {
    fetch("/api/v1/traces?limit=50")
      .then((r) => r.json())
      .then((d) => setTraces(d.traces || []))
      .catch(() => {});
  }, []);

  /* select a trace */
  const selectTrace = useCallback(async (tid: string) => {
    setSelected(tid);
    setLoading(true);
    try {
      const [detail, seq] = await Promise.all([
        fetch(`/api/v1/traces/${tid}`).then((r) => r.json()),
        fetch(`/api/v1/traces/${tid}/sequence`).then((r) => r.json()),
      ]);
      setSpans(detail.spans || []);
      setMermaidCode(seq.mermaid || "");
    } catch {
      setSpans([]);
      setMermaidCode("");
    } finally {
      setLoading(false);
    }
  }, []);

  /* filtered traces */
  const filteredTraces = useMemo(() => {
    switch (filter) {
      case "errors":
        return traces.filter((t) => t.status === "error");
      case "zero":
        return traces.filter((t) => t.n_results === 0);
      case "slow":
        return traces.filter((t) => t.latency_ms > 1000);
      default:
        return traces;
    }
  }, [traces, filter]);

  /* selected trace object */
  const selectedTrace = useMemo(
    () => traces.find((t) => t.trace_id === selected) ?? null,
    [traces, selected],
  );

  /* max latency for waterfall bar */
  const maxSpanLatency = useMemo(
    () => Math.max(...spans.map((s) => s.latency_ms), 1),
    [spans],
  );

  /* row background */
  function rowBg(t: Trace): string | undefined {
    if (t.trace_id === selected) return "hsl(204, 86%, 93%)";
    if (t.status === "error") return "hsl(348, 100%, 95%)";
    if (t.n_results === 0) return "hsl(48, 100%, 93%)";
    return undefined;
  }

  const filterButtons: { key: FilterKey; label: string }[] = [
    { key: "all", label: "All" },
    { key: "errors", label: "Errors" },
    { key: "zero", label: "Zero Results" },
    { key: "slow", label: "Slow (>1s)" },
  ];

  return (
    <div style={{ maxWidth: "1400px", margin: "0 auto", padding: "1rem" }}>
      <h2 className="title is-5 mb-3">Traces</h2>

      <div className="columns">
        {/* ── left: trace list ─────────────────────────────── */}
        <div className="column is-5">
          {/* filter buttons */}
          <div className="buttons are-small mb-2">
            {filterButtons.map((f) => (
              <button
                key={f.key}
                className={`button is-small ${filter === f.key ? "is-link" : "is-light"}`}
                onClick={() => setFilter(f.key)}
              >
                {f.label}
              </button>
            ))}
          </div>

          <p className="is-size-7 has-text-grey mb-2">
            {filteredTraces.length} trace{filteredTraces.length !== 1 ? "s" : ""}
            {filter !== "all" ? ` (filtered from ${traces.length})` : ""}
          </p>

          <div
            className="table-container"
            style={{ maxHeight: "72vh", overflow: "auto" }}
          >
            <table className="table is-narrow is-striped is-fullwidth is-hoverable">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Query</th>
                  <th>Mode</th>
                  <th>Status</th>
                  <th>ms</th>
                  <th>Results</th>
                </tr>
              </thead>
              <tbody>
                {filteredTraces.map((t) => (
                  <tr
                    key={t.trace_id}
                    onClick={() => selectTrace(t.trace_id)}
                    style={{ cursor: "pointer", background: rowBg(t) }}
                  >
                    <td className="is-size-7" style={{ whiteSpace: "nowrap" }}>
                      {new Date(t.ts * 1000).toLocaleTimeString()}
                    </td>
                    <td
                      className="is-size-7"
                      title={t.query_raw}
                      style={{ maxWidth: "180px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                    >
                      {truncate(t.query_raw || "", 50)}
                    </td>
                    <td>
                      <span className="tag is-small is-light">
                        {t.mode || t.endpoint || "?"}
                      </span>
                    </td>
                    <td>
                      <span
                        className={`tag is-small ${
                          t.status === "error"
                            ? "is-danger"
                            : t.status === "ok"
                              ? "is-success"
                              : "is-light"
                        }`}
                      >
                        {t.status || "-"}
                      </span>
                    </td>
                    <td className="is-size-7">
                      {t.latency_ms?.toFixed(0) ?? "-"}
                    </td>
                    <td className="is-size-7">{t.n_results ?? "-"}</td>
                  </tr>
                ))}
                {filteredTraces.length === 0 && (
                  <tr>
                    <td colSpan={6} className="has-text-centered has-text-grey">
                      No traces match this filter
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* ── right: trace detail ──────────────────────────── */}
        <div className="column is-7">
          {loading && (
            <progress className="progress is-small is-primary" max="100" />
          )}

          {selected && !loading && selectedTrace && (
            <>
              {/* trace detail header */}
              <div
                className="box py-3 px-4 mb-3"
                style={{ background: "hsl(0, 0%, 96%)" }}
              >
                <div className="columns is-multiline is-mobile mb-0">
                  <div className="column is-12 py-1">
                    <span className="is-size-7 has-text-grey">Trace ID </span>
                    <code className="is-size-7">{selectedTrace.trace_id}</code>
                  </div>
                  <div className="column is-4 py-1">
                    <span className="is-size-7 has-text-grey">Endpoint </span>
                    <span className="is-size-7 has-text-weight-medium">
                      {selectedTrace.endpoint || "-"}
                    </span>
                  </div>
                  <div className="column is-3 py-1">
                    <span className="is-size-7 has-text-grey">Mode </span>
                    <span className="tag is-small is-info is-light">
                      {selectedTrace.mode || "-"}
                    </span>
                  </div>
                  <div className="column is-2 py-1">
                    <span className="is-size-7 has-text-grey">Status </span>
                    <span
                      className={`tag is-small ${
                        selectedTrace.status === "error"
                          ? "is-danger"
                          : selectedTrace.status === "ok"
                            ? "is-success"
                            : "is-warning"
                      }`}
                    >
                      {selectedTrace.status || "-"}
                    </span>
                  </div>
                  <div className="column is-3 py-1">
                    <span className="is-size-7 has-text-grey">Latency </span>
                    <span className="is-size-7 has-text-weight-bold">
                      {selectedTrace.latency_ms?.toFixed(0)} ms
                    </span>
                  </div>
                  <div className="column is-3 py-1">
                    <span className="is-size-7 has-text-grey">Results </span>
                    <span className="is-size-7 has-text-weight-bold">
                      {selectedTrace.n_results ?? 0}
                    </span>
                  </div>
                </div>
              </div>

              {/* spans */}
              <p className="subtitle is-6 mb-2">
                Spans ({spans.length})
              </p>
              <div style={{ maxHeight: "45vh", overflow: "auto" }}>
                {spans.map((s) => (
                  <SpanCard
                    key={s.span_id}
                    span={s}
                    maxLatency={maxSpanLatency}
                  />
                ))}
                {spans.length === 0 && (
                  <p className="has-text-grey is-size-7">No spans recorded</p>
                )}
              </div>

              {/* mermaid diagram */}
              {mermaidCode && (
                <>
                  <p className="subtitle is-6 mt-4 mb-2">Sequence Diagram</p>
                  <MermaidDiagram code={mermaidCode} />
                </>
              )}
            </>
          )}

          {!selected && !loading && (
            <p className="has-text-grey has-text-centered mt-6">
              Select a trace to view its spans and sequence diagram
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

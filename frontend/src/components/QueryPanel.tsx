import { useEffect, useMemo, useState } from "react";
import type { QueryItem, SearchConfig as Config, SearchFilters } from "../types";
import { DEFAULT_CONFIG } from "../types";
import * as api from "../api/client";
import QueryBox from "./QueryBox";
import SearchConfigPanel from "./SearchConfig";
import SketchCanvas from "./SketchCanvas";

interface Props {
  onSearch: (
    queries: QueryItem[],
    config: Config,
    filters: SearchFilters,
    imageQuery?: string
  ) => void;
  loading: boolean;
}

const MODEL_INFO: Record<string, { label: string; full: string }> = {
  siglip2: { label: "SigLIP2", full: "google/siglip2-base-patch16-512" },
  clip: { label: "OpenCLIP", full: "open_clip ViT-B-16 + TSBIR" },
};

export default function QueryPanel({ onSearch, loading }: Props) {
  const [queries, setQueries] = useState<QueryItem[]>([
    { text: "", lang: "en" },
  ]);
  const [config, setConfig] = useState<Config>(DEFAULT_CONFIG);
  const [filters, setFilters] = useState<SearchFilters>({ objects: [], ocr: [] });
  const [showConfig, setShowConfig] = useState(false);
  const [showSketch, setShowSketch] = useState(false);
  const [sketchData, setSketchData] = useState<string | null>(null);
  const [objectClasses, setObjectClasses] = useState<string[]>([]);

  useEffect(() => {
    api.getObjectClasses().then((res) => setObjectClasses(res.classes)).catch(() => {});
  }, []);

  useEffect(() => {
    if (sketchData) {
      setConfig((prev) => {
        if (prev.encoder_mode !== "clip") {
          api.switchEncoder("clip").catch(() => {});
        }
        return { ...prev, encoder_mode: "clip" };
      });
    }
  }, [sketchData]);

  const updateQuery = (index: number, partial: Partial<QueryItem>) => {
    setQueries((prev) =>
      prev.map((q, i) => (i === index ? { ...q, ...partial } : q))
    );
  };

  const addQuery = () => {
    setQueries((prev) => [...prev, { text: "", lang: "en" }]);
  };

  const removeQuery = (index: number) => {
    setQueries((prev) => prev.filter((_, i) => i !== index));
  };

  const effectiveMode = sketchData ? "clip" : config.encoder_mode;

  const strategy = useMemo(() => {
    const textCount = queries.filter((q) => q.text.trim()).length;
    const sketchCount = sketchData ? 1 : 0;
    const totalVectors = textCount + sketchCount;

    if (totalVectors === 0) return null;

    const mode = totalVectors === 1 ? "ANN" : "RRF";
    const hasText = textCount > 0;
    const effectiveRerank = config.rerank && hasText;

    return { mode, effectiveRerank, modelLabel: MODEL_INFO[effectiveMode].label };
  }, [queries, sketchData, effectiveMode, config.rerank]);

  const handleSearch = () => {
    const nonEmpty = queries.filter((q) => q.text.trim());
    if (nonEmpty.length === 0 && !sketchData) return;

    const effectiveConfig = { ...config };
    if (sketchData) effectiveConfig.encoder_mode = "clip";
    if (nonEmpty.length === 0 && sketchData) effectiveConfig.rerank = false;

    onSearch(
      nonEmpty.length > 0 ? nonEmpty : [{ text: "", lang: "en" }],
      effectiveConfig,
      filters,
      sketchData || undefined
    );
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      handleSearch();
    }
  };

  return (
    <div className="query-panel p-3" onKeyDown={handleKeyDown}>
      <h2 className="title is-5 mb-3">Search</h2>

      {queries.map((q, i) => (
        <QueryBox
          key={i}
          index={i}
          text={q.text}
          lang={q.lang}
          canRemove={queries.length > 1}
          onTextChange={(text) => updateQuery(i, { text })}
          onLangChange={(lang) => updateQuery(i, { lang })}
          onRemove={() => removeQuery(i)}
        />
      ))}

      <div className="is-flex" style={{ gap: "0.5rem" }}>
        <button className="button is-small is-light" onClick={addQuery}>
          + Add Query
        </button>
        <button
          className={`button is-small ${showSketch ? "is-link" : "is-light"}`}
          onClick={() => setShowSketch(!showSketch)}
        >
          {showSketch ? "Hide Sketch" : "Sketch"}
        </button>
      </div>

      {showSketch && (
        <div className="box mt-3 mb-3 p-3">
          <div className="is-flex is-justify-content-space-between is-align-items-center mb-2">
            <span className="tag is-warning is-light">Sketch (TSBIR)</span>
            {sketchData && <span className="tag is-success is-light is-small">Active</span>}
          </div>
          <SketchCanvas onChange={setSketchData} />
        </div>
      )}

      {/* ── Encoder ── */}
      <div className="box mt-3 p-3">
        <div className="is-flex is-align-items-center mb-2" style={{ gap: "0.5rem" }}>
          <p className="subtitle is-6 mb-0">Encoder</p>
          <span className="tag is-light is-small">{MODEL_INFO[effectiveMode].label}</span>
        </div>
        <div className="field mb-2">
          <div className="control">
            <div className="select is-small is-fullwidth">
              <select
                value={effectiveMode}
                disabled={!!sketchData}
                onChange={(e) => {
                  const newMode = e.target.value as Config["encoder_mode"];
                  if (newMode === config.encoder_mode) return;
                  if (!window.confirm(`Switch to ${MODEL_INFO[newMode].label}? This will unload the current model from memory.`)) return;
                  api.switchEncoder(newMode).catch(() => {});
                  setConfig({ ...config, encoder_mode: newMode });
                }}
              >
                <option value="siglip2">google/siglip2-base-patch16-512</option>
                <option value="clip">open_clip ViT-B-16 + TSBIR</option>
              </select>
            </div>
          </div>
        </div>
        <div className="field mb-0">
          <label className="label is-small mb-1">Metric</label>
          <div className="control">
            <div className="select is-small is-fullwidth">
              <select
                value={config.metric_type}
                onChange={(e) =>
                  setConfig({ ...config, metric_type: e.target.value as Config["metric_type"] })
                }
              >
                <option value="COSINE">Cosine</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* ── Filters ── */}
      <div className="box p-3">
        <p className="subtitle is-6 mb-2">Filters</p>
        <div className="field mb-2">
          <div className="is-flex is-align-items-center mb-1" style={{ gap: "0.4rem" }}>
            <label className="label is-small mb-0">Objects</label>
            <span className="tag is-light is-small">faster_rcnn/inception_resnet_v2</span>
          </div>
          <div className="control">
            <div className="select is-small is-fullwidth">
              <select
                value=""
                onChange={(e) => {
                  const val = e.target.value;
                  if (val && !filters.objects.includes(val)) {
                    setFilters({ ...filters, objects: [...filters.objects, val] });
                  }
                }}
              >
                <option value="">Select object class...</option>
                {objectClasses
                  .filter((cls) => !filters.objects.includes(cls))
                  .map((cls) => (
                    <option key={cls} value={cls}>{cls}</option>
                  ))}
              </select>
            </div>
          </div>
          {filters.objects.length > 0 && (
            <div className="tags mt-1">
              {filters.objects.map((obj) => (
                <span key={obj} className="tag is-info is-light">
                  {obj}
                  <button
                    className="delete is-small"
                    onClick={() =>
                      setFilters({ ...filters, objects: filters.objects.filter((o) => o !== obj) })
                    }
                  />
                </span>
              ))}
            </div>
          )}
        </div>
        <div className="field mb-0">
          <div className="is-flex is-align-items-center mb-1" style={{ gap: "0.4rem" }}>
            <label className="label is-small mb-0">OCR</label>
            <span className="tag is-light is-small">microsoft/Florence-2-base</span>
          </div>
          <div className="control">
            <input
              type="text"
              className="input is-small"
              placeholder="HTV9, FANA"
              value={filters.ocr.join(", ")}
              onChange={(e) => {
                const vals = e.target.value
                  .split(",")
                  .map((s) => s.trim())
                  .filter(Boolean);
                setFilters({ ...filters, ocr: vals });
              }}
            />
          </div>
        </div>
      </div>

      {/* ── Reranking ── */}
      <div className="box p-3">
        <div className="is-flex is-align-items-center mb-2" style={{ gap: "0.5rem" }}>
          <p className="subtitle is-6 mb-0">Reranking</p>
          <span className="tag is-light is-small">BAAI/bge-reranker-v2-m3</span>
        </div>
        <label className="checkbox is-size-7">
          <input
            type="checkbox"
            checked={config.rerank}
            onChange={(e) => setConfig({ ...config, rerank: e.target.checked })}
          />{" "}
          Enable
        </label>
        {config.rerank && (
          <div className="field mt-2 mb-0">
            <label className="label is-small mb-0">
              Candidates = {config.rerank_candidates}
            </label>
            <div className="control">
              <input
                type="range" min={10} max={200} step={10}
                value={config.rerank_candidates}
                onChange={(e) => setConfig({ ...config, rerank_candidates: Number(e.target.value) })}
                style={{ width: "100%" }}
              />
            </div>
          </div>
        )}
      </div>

      {/* ── Strategy indicator ── */}
      {strategy && (
        <div className="is-flex is-align-items-center mt-2 mb-2" style={{ gap: "0.25rem" }}>
          <span className="tag is-small is-light">{strategy.modelLabel}</span>
          <span className={`tag is-small ${strategy.mode === "ANN" ? "is-info is-light" : "is-warning is-light"}`}>
            {strategy.mode}
          </span>
          {strategy.effectiveRerank && (
            <span className="tag is-small is-success is-light">Rerank</span>
          )}
        </div>
      )}

      <button
        className={`button is-primary is-fullwidth mt-2 ${loading ? "is-loading" : ""}`}
        onClick={handleSearch}
        disabled={loading}
      >
        Search
      </button>

      <div className="mt-3">
        <button
          className="button is-small is-ghost"
          onClick={() => setShowConfig(!showConfig)}
        >
          {showConfig ? "Hide" : "Show"} Advanced Config
        </button>
        {showConfig && (
          <SearchConfigPanel
            config={config}
            queryCount={queries.length}
            onChange={setConfig}
          />
        )}
      </div>
    </div>
  );
}

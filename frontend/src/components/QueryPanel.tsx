import { useEffect, useMemo, useState } from "react";
import type {
  QueryItem,
  SearchConfig as Config,
  SearchFilters,
  ParsedQuery,
  V1SearchRequest,
} from "../types";
import { DEFAULT_CONFIG } from "../types";
import * as api from "../api/client";
import QueryBox from "./QueryBox";
import SearchConfigPanel from "./SearchConfig";
import SketchCanvas from "./SketchCanvas";
import ParsedQueryEditor from "./ParsedQueryEditor";
import FlagsPanel from "./FlagsPanel";

interface Props {
  onSearch: (
    queries: QueryItem[],
    config: Config,
    filters: SearchFilters,
    imageQuery?: string
  ) => void;
  onV1Search?: (req: V1SearchRequest) => void;
  loading: boolean;
  upvoteIds?: string[];
  downvoteIds?: string[];
}

export default function QueryPanel({
  onSearch,
  onV1Search,
  loading,
  upvoteIds = [],
  downvoteIds = [],
}: Props) {
  const [queries, setQueries] = useState<QueryItem[]>([
    { text: "", lang: "en" },
  ]);
  const [config, setConfig] = useState<Config>(DEFAULT_CONFIG);
  const [filters, setFilters] = useState<SearchFilters>({
    objects: [],
    ocr: [],
  });
  const [showConfig, setShowConfig] = useState(false);
  const [showSketch, setShowSketch] = useState(false);
  const [sketchData, setSketchData] = useState<string | null>(null);
  const [objectClasses, setObjectClasses] = useState<string[]>([]);

  // v1 state
  const [parsed, setParsed] = useState<ParsedQuery | null>(null);
  const [parsing, setParsing] = useState(false);
  const [parseError, setParseError] = useState<string | null>(null);
  const [showFlags, setShowFlags] = useState(false);
  const [flagOverrides, setFlagOverrides] = useState<Record<string, unknown>>(
    {}
  );

  useEffect(() => {
    api
      .getObjectClasses()
      .then((res) => setObjectClasses(res.classes))
      .catch(() => {});
  }, []);

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

  const strategy = useMemo(() => {
    const textCount = queries.filter((q) => q.text.trim()).length;
    const sketchCount = sketchData ? 1 : 0;
    const totalVectors = textCount + sketchCount;
    if (totalVectors === 0) return null;
    const mode = totalVectors === 1 ? "ANN" : "RRF";
    const encoder = sketchData ? "SigLIP2 + TSBIR" : "SigLIP2";
    return { mode, encoder };
  }, [queries, sketchData]);

  const handleParse = async () => {
    const text = queries[0]?.text?.trim();
    if (!text) return;
    setParsing(true);
    setParseError(null);
    try {
      const resp = await api.parseQuery(
        text,
        queries[0].lang === "vi" ? "vi" : "auto",
        Object.keys(flagOverrides).length > 0 ? flagOverrides : undefined
      );
      const withEnabled = {
        ...resp.parsed,
        events: resp.parsed.events.map((e) => ({ ...e, enabled: true })),
      };
      setParsed(withEnabled);
    } catch (e) {
      setParseError(e instanceof Error ? e.message : "Parse failed");
    } finally {
      setParsing(false);
    }
  };

  const handleV1Search = () => {
    if (!onV1Search || !parsed) return;
    const enabledEvents = parsed.events.filter((e) => e.enabled);
    if (enabledEvents.length === 0) return;
    const searchParsed = { ...parsed, events: enabledEvents };
    onV1Search({
      parsed: searchParsed,
      upvote_kf_ids: upvoteIds.length > 0 ? upvoteIds : undefined,
      downvote_kf_ids: downvoteIds.length > 0 ? downvoteIds : undefined,
      flags: Object.keys(flagOverrides).length > 0 ? flagOverrides : undefined,
    });
  };

  const handleLegacySearch = () => {
    const nonEmpty = queries.filter((q) => q.text.trim());
    if (nonEmpty.length === 0 && !sketchData) return;
    onSearch(
      nonEmpty.length > 0 ? nonEmpty : [{ text: "", lang: "en" }],
      config,
      filters,
      sketchData || undefined
    );
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      if (parsed && onV1Search) {
        handleV1Search();
      } else {
        handleLegacySearch();
      }
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
          onTextChange={(text) => {
            updateQuery(i, { text });
            setParsed(null);
          }}
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
        <button
          className={`button is-small ${showFlags ? "is-link is-light" : "is-light"}`}
          onClick={() => setShowFlags(!showFlags)}
        >
          Flags
        </button>
      </div>

      {showSketch && (
        <div className="box mt-3 mb-3 p-3">
          <div className="is-flex is-justify-content-space-between is-align-items-center mb-2">
            <span className="tag is-warning is-light">Sketch (TSBIR)</span>
            {sketchData && (
              <span className="tag is-success is-light is-small">Active</span>
            )}
          </div>
          <SketchCanvas onChange={setSketchData} />
        </div>
      )}

      {showFlags && (
        <div className="box mt-2 mb-2 p-3">
          <p className="subtitle is-6 mb-2">Feature Flags</p>
          <FlagsPanel overrides={flagOverrides} onChange={setFlagOverrides} />
        </div>
      )}

      {/* Parse button */}
      {onV1Search && (
        <div className="mt-2 mb-2">
          <button
            className={`button is-small is-info is-outlined is-fullwidth ${parsing ? "is-loading" : ""}`}
            onClick={handleParse}
            disabled={parsing || !queries[0]?.text?.trim()}
          >
            Parse Query (LLM)
          </button>
          {parseError && (
            <p className="help is-danger mt-1">{parseError}</p>
          )}
        </div>
      )}

      {/* Parsed query editor */}
      {parsed && (
        <div className="box mt-2 mb-2 p-3">
          <ParsedQueryEditor parsed={parsed} onChange={setParsed} />
        </div>
      )}

      {/* Filters (legacy) */}
      {!parsed && (
        <div className="box mt-3 p-3">
          <p className="subtitle is-6 mb-2">Filters</p>
          <div className="field mb-2">
            <div
              className="is-flex is-align-items-center mb-1"
              style={{ gap: "0.4rem" }}
            >
              <label className="label is-small mb-0">Objects</label>
              <span className="tag is-light is-small">YOLO11 COCO</span>
            </div>
            <div className="control">
              <div className="select is-small is-fullwidth">
                <select
                  value=""
                  onChange={(e) => {
                    const val = e.target.value;
                    if (val && !filters.objects.includes(val)) {
                      setFilters({
                        ...filters,
                        objects: [...filters.objects, val],
                      });
                    }
                  }}
                >
                  <option value="">Select object class...</option>
                  {objectClasses
                    .filter((cls) => !filters.objects.includes(cls))
                    .map((cls) => (
                      <option key={cls} value={cls}>
                        {cls}
                      </option>
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
                        setFilters({
                          ...filters,
                          objects: filters.objects.filter((o) => o !== obj),
                        })
                      }
                    />
                  </span>
                ))}
              </div>
            )}
          </div>
          <div className="field mb-0">
            <div
              className="is-flex is-align-items-center mb-1"
              style={{ gap: "0.4rem" }}
            >
              <label className="label is-small mb-0">OCR</label>
              <span className="tag is-light is-small">Florence-2</span>
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
      )}

      {strategy && (
        <div
          className="is-flex is-align-items-center mt-2 mb-2"
          style={{ gap: "0.25rem" }}
        >
          <span className="tag is-small is-light">{strategy.encoder}</span>
          <span
            className={`tag is-small ${strategy.mode === "ANN" ? "is-info is-light" : "is-warning is-light"}`}
          >
            {strategy.mode}
          </span>
          {(upvoteIds.length > 0 || downvoteIds.length > 0) && (
            <span className="tag is-small is-success is-light">
              +{upvoteIds.length} -{downvoteIds.length}
            </span>
          )}
        </div>
      )}

      {/* Search buttons */}
      {parsed && onV1Search ? (
        <button
          className={`button is-primary is-fullwidth mt-2 ${loading ? "is-loading" : ""}`}
          onClick={handleV1Search}
          disabled={loading}
        >
          Search (v1)
        </button>
      ) : (
        <button
          className={`button is-primary is-fullwidth mt-2 ${loading ? "is-loading" : ""}`}
          onClick={handleLegacySearch}
          disabled={loading}
        >
          Search
        </button>
      )}

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

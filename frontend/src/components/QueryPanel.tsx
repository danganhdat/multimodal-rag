import { useState } from "react";
import type { QueryItem, SearchConfig as Config } from "../types";
import { DEFAULT_CONFIG } from "../types";
import QueryBox from "./QueryBox";
import SearchConfigPanel from "./SearchConfig";

interface Props {
  onSearch: (
    queries: QueryItem[],
    config: Config,
    filters: { objects: string[] }
  ) => void;
  loading: boolean;
}

export default function QueryPanel({ onSearch, loading }: Props) {
  const [queries, setQueries] = useState<QueryItem[]>([
    { text: "", lang: "en" },
  ]);
  const [config, setConfig] = useState<Config>(DEFAULT_CONFIG);
  const [objectFilter, setObjectFilter] = useState("");
  const [showConfig, setShowConfig] = useState(false);

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

  const handleSearch = () => {
    const nonEmpty = queries.filter((q) => q.text.trim());
    if (nonEmpty.length === 0) return;
    const objects = objectFilter
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    onSearch(nonEmpty, config, { objects });
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
        <button className="button is-small is-outlined" onClick={addQuery}>
          + Add Query
        </button>
      </div>

      <div className="field mt-3">
        <label className="label is-small">Object filter (comma-separated)</label>
        <div className="control">
          <input
            type="text"
            className="input is-small"
            placeholder="Car, Person, Building..."
            value={objectFilter}
            onChange={(e) => setObjectFilter(e.target.value)}
          />
        </div>
      </div>

      <button
        className={`button is-primary is-fullwidth mt-3 ${loading ? "is-loading" : ""}`}
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
          {showConfig ? "Hide" : "Show"} Config
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

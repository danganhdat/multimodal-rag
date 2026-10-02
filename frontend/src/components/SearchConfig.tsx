import type { SearchConfig as Config } from "../types";

interface Props {
  config: Config;
  queryCount: number;
  onChange: (config: Config) => void;
}

export default function SearchConfigPanel({ config, queryCount, onChange }: Props) {
  const update = (partial: Partial<Config>) =>
    onChange({ ...config, ...partial });

  return (
    <div className="box">
      <p className="subtitle is-6 mb-3">Search Config</p>

      <div className="field">
        <label className="label is-small">Ranker</label>
        <div className="control">
          <div className="select is-small is-fullwidth">
            <select
              value={config.ranker}
              onChange={(e) => update({ ranker: e.target.value as Config["ranker"] })}
            >
              <option value="rrf">RRF (Reciprocal Rank Fusion)</option>
              <option value="weighted">Weighted</option>
            </select>
          </div>
        </div>
      </div>

      {config.ranker === "rrf" && (
        <div className="field">
          <label className="label is-small">RRF k = {config.rrf_k}</label>
          <div className="control">
            <input
              type="range"
              min={1}
              max={200}
              value={config.rrf_k}
              onChange={(e) => update({ rrf_k: Number(e.target.value) })}
              className="slider is-fullwidth"
              style={{ width: "100%" }}
            />
          </div>
        </div>
      )}

      {config.ranker === "weighted" && queryCount > 1 && (
        <div className="field">
          <label className="label is-small">Weights</label>
          {Array.from({ length: queryCount }).map((_, i) => (
            <div key={i} className="field has-addons mb-1">
              <p className="control">
                <span className="button is-small is-static">Q{i + 1}</span>
              </p>
              <p className="control is-expanded">
                <input
                  type="number"
                  className="input is-small"
                  min={0}
                  max={1}
                  step={0.1}
                  value={config.weights[i] ?? 0.5}
                  onChange={(e) => {
                    const w = [...config.weights];
                    while (w.length < queryCount) w.push(0.5);
                    w[i] = Number(e.target.value);
                    update({ weights: w });
                  }}
                />
              </p>
            </div>
          ))}
        </div>
      )}

      <div className="field">
        <label className="label is-small">Metric</label>
        <div className="control">
          <div className="select is-small is-fullwidth">
            <select
              value={config.metric_type}
              onChange={(e) => update({ metric_type: e.target.value as Config["metric_type"] })}
            >
              <option value="COSINE">COSINE</option>
              <option value="IP">IP (Inner Product)</option>
              <option value="L2">L2 (Euclidean)</option>
            </select>
          </div>
        </div>
      </div>

      <div className="field">
        <label className="label is-small">Top-K = {config.top_k}</label>
        <div className="control">
          <input
            type="range"
            min={10}
            max={200}
            step={10}
            value={config.top_k}
            onChange={(e) => update({ top_k: Number(e.target.value) })}
            style={{ width: "100%" }}
          />
        </div>
      </div>

      <div className="field">
        <label className="label is-small">Neighbor window = {config.neighbor_window}</label>
        <div className="control">
          <input
            type="range"
            min={1}
            max={20}
            value={config.neighbor_window}
            onChange={(e) => update({ neighbor_window: Number(e.target.value) })}
            style={{ width: "100%" }}
          />
        </div>
      </div>

      <div className="field">
        <label className="checkbox is-size-7">
          <input
            type="checkbox"
            checked={config.group_by_video}
            onChange={(e) => update({ group_by_video: e.target.checked })}
          />{" "}
          Group by video
        </label>
      </div>

      <div className="field">
        <label className="checkbox is-size-7">
          <input
            type="checkbox"
            checked={config.use_client_merge}
            onChange={(e) => update({ use_client_merge: e.target.checked })}
          />{" "}
          Client-side merge (ablation)
        </label>
      </div>

      <hr className="my-3" />

      <div className="field">
        <label className="checkbox is-size-7">
          <input
            type="checkbox"
            checked={config.rerank}
            onChange={(e) => update({ rerank: e.target.checked })}
          />{" "}
          Rerank (BGE-reranker-v2-m3)
        </label>
      </div>

      {config.rerank && (
        <div className="field">
          <label className="label is-small">
            Rerank candidates = {config.rerank_candidates}
          </label>
          <div className="control">
            <input
              type="range"
              min={50}
              max={500}
              step={50}
              value={config.rerank_candidates}
              onChange={(e) =>
                update({ rerank_candidates: Number(e.target.value) })
              }
              style={{ width: "100%" }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

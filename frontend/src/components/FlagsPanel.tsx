import { useEffect, useState } from "react";
import * as api from "../api/client";
import type { V1ConfigResponse } from "../types";

interface Props {
  overrides: Record<string, unknown>;
  onChange: (overrides: Record<string, unknown>) => void;
}

export default function FlagsPanel({ overrides, onChange }: Props) {
  const [config, setConfig] = useState<V1ConfigResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .v1Config()
      .then(setConfig)
      .catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <div className="notification is-danger is-light py-2 px-3 is-size-7">
        Config: {error}
      </div>
    );
  }
  if (!config) return <p className="is-size-7">Loading config...</p>;

  const flags = config.flags as Record<string, Record<string, unknown>>;
  const status = config.status;

  const toggle = (section: string, key: string, value: boolean) => {
    const prev = (overrides[section] as Record<string, unknown>) || {};
    onChange({ ...overrides, [section]: { ...prev, [key]: value } });
  };

  const setVal = (section: string, key: string, value: unknown) => {
    const prev = (overrides[section] as Record<string, unknown>) || {};
    onChange({ ...overrides, [section]: { ...prev, [key]: value } });
  };

  const get = (section: string, key: string) => {
    const ov = overrides[section] as Record<string, unknown> | undefined;
    if (ov && key in ov) return ov[key];
    return (flags[section] as Record<string, unknown>)?.[key];
  };

  return (
    <div className="flags-panel">
      {/* Status badges */}
      <div className="is-flex mb-2" style={{ gap: "0.25rem", flexWrap: "wrap" }}>
        <span className={`tag is-small ${status.milvus ? "is-success" : "is-danger"} is-light`}>
          Milvus
        </span>
        <span className={`tag is-small ${status.bm25 ? "is-success" : "is-warning"} is-light`}>
          BM25
        </span>
        <span className={`tag is-small ${status.llm ? "is-success" : "is-warning"} is-light`}>
          LLM
        </span>
        <span className={`tag is-small ${status.vlm ? "is-success" : "is-warning"} is-light`}>
          VLM
        </span>
        <span className="tag is-small is-light">{config.config_id}</span>
      </div>

      {/* QE */}
      <FlagRow
        label="Query Enhancement"
        checked={!!get("qe", "enabled")}
        onChange={(v) => toggle("qe", "enabled", v)}
      />

      {/* Search */}
      <FlagRow
        label="Dense search"
        checked={!!get("search", "dense")}
        onChange={(v) => toggle("search", "dense", v)}
      />
      <FlagRow
        label="BM25 (OCR)"
        checked={!!get("search", "bm25")}
        onChange={(v) => toggle("search", "bm25", v)}
        disabled={!status.bm25}
        tooltip={!status.bm25 ? "No OCR data in Milvus" : undefined}
      />

      {/* Rerank */}
      <div className="field mb-1">
        <label className="label is-small mb-0">Rerank</label>
        <div className="select is-small">
          <select
            value={String(get("rerank", "method") ?? "superglobal")}
            onChange={(e) => setVal("rerank", "method", e.target.value)}
          >
            <option value="none">none</option>
            <option value="superglobal">superglobal</option>
            <option value="mllm">mllm</option>
            <option value="sg+mllm">sg+mllm</option>
            <option value="vlm">vlm</option>
          </select>
        </div>
      </div>

      {/* Temporal */}
      <FlagRow
        label="Temporal search"
        checked={!!get("temporal", "enabled")}
        onChange={(v) => toggle("temporal", "enabled", v)}
      />

      {Object.keys(overrides).length > 0 && (
        <button
          className="button is-small is-light mt-1"
          onClick={() => onChange({})}
        >
          Reset flags
        </button>
      )}
    </div>
  );
}

function FlagRow({
  label,
  checked,
  onChange,
  disabled,
  tooltip,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
  tooltip?: string;
}) {
  return (
    <label
      className="checkbox is-size-7 is-block mb-1"
      title={tooltip}
      style={disabled ? { opacity: 0.5 } : undefined}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        disabled={disabled}
      />{" "}
      {label}
      {disabled && tooltip && (
        <span className="has-text-grey ml-1">({tooltip})</span>
      )}
    </label>
  );
}

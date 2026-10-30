import { useState } from "react";
import type { AgentEvent } from "../api/sse";

interface Props {
  events: AgentEvent[];
  onEnlargeImage?: (url: string) => void;
}

export function kfIdToImageUrl(kfId: string): string {
  const [video, n] = kfId.split("/");
  return `/keyframes/${video}/${n}.jpg`;
}

interface ParsedSearchResult {
  kfId: string;
  score: number;
  objects: string;
  ocr: string;
}

export function parseSearchObservation(content: string): ParsedSearchResult[] {
  const results: ParsedSearchResult[] = [];
  const lines = content.split("\n");
  for (const line of lines) {
    const match = line.match(/^\d+\.\s+(L\d+_V\d+\/\d{3,})\s+\(score=([\d.]+)\)/);
    if (match) {
      const objMatch = line.match(/objects=\[([^\]]*)\]/);
      const ocrMatch = line.match(/ocr=<data>(.*?)<\/data>/);
      results.push({
        kfId: match[1],
        score: parseFloat(match[2]),
        objects: objMatch ? objMatch[1] : "",
        ocr: ocrMatch ? ocrMatch[1] : "",
      });
    }
  }
  return results;
}

interface ParsedLookResult {
  frameIds: string[];
  observation: string;
}

export function parseLookObservation(content: string): ParsedLookResult | null {
  try {
    const data = JSON.parse(content);
    return {
      frameIds: data.frame_ids || [],
      observation: data.observation || "",
    };
  } catch {
    return null;
  }
}

function SearchResultStrip({ results, onEnlargeImage }: { results: ParsedSearchResult[]; onEnlargeImage?: (url: string) => void }) {
  const top = results.slice(0, 8);
  return (
    <div className="is-flex" style={{ gap: "4px", overflowX: "auto", padding: "4px 0" }}>
      {top.map((r) => {
        const url = kfIdToImageUrl(r.kfId);
        return (
          <div
            key={r.kfId}
            style={{ minWidth: "120px", maxWidth: "160px", cursor: onEnlargeImage ? "pointer" : "default" }}
            onClick={() => onEnlargeImage?.(url)}
          >
            <figure className="image is-16by9" style={{ margin: 0 }}>
              <img
                src={url}
                alt={r.kfId}
                loading="lazy"
                style={{ objectFit: "cover", borderRadius: "3px" }}
              />
            </figure>
            <div className="is-size-7 has-text-grey" style={{ lineHeight: 1.2, marginTop: "2px" }}>
              <span className="has-text-weight-bold">{r.kfId}</span>
              <span className="ml-1">{r.score.toFixed(3)}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function LookResultDisplay({ data, onEnlargeImage }: { data: ParsedLookResult; onEnlargeImage?: (url: string) => void }) {
  return (
    <div>
      <div className="is-flex" style={{ gap: "4px", marginBottom: "4px" }}>
        {data.frameIds.map((kfId) => {
          const url = kfIdToImageUrl(kfId);
          return (
            <figure
              key={kfId}
              className="image"
              style={{ width: "100px", margin: 0, cursor: onEnlargeImage ? "pointer" : "default" }}
              onClick={() => onEnlargeImage?.(url)}
            >
              <img
                src={url}
                alt={kfId}
                loading="lazy"
                style={{ objectFit: "cover", borderRadius: "3px", maxHeight: "70px" }}
              />
            </figure>
          );
        })}
      </div>
      <p className="is-size-7" style={{ whiteSpace: "pre-wrap", maxHeight: "150px", overflow: "auto" }}>
        {data.observation.slice(0, 800)}
      </p>
    </div>
  );
}

export default function AgentTimeline({ events, onEnlargeImage }: Props) {
  if (events.length === 0) return null;

  return (
    <div className="agent-timeline">
      {events.map((event, i) => (
        <TimelineItem key={i} event={event} onEnlargeImage={onEnlargeImage} />
      ))}
    </div>
  );
}

function TimelineItem({ event, onEnlargeImage }: { event: AgentEvent; onEnlargeImage?: (url: string) => void }) {
  const [expanded, setExpanded] = useState(false);

  switch (event.type) {
    case "thought":
      return (
        <div className="box py-2 px-3 mb-1" style={{ borderLeft: "3px solid hsl(204, 86%, 53%)" }}>
          <span className="tag is-info is-light is-small mr-1">Thought</span>
          <span className="is-size-7">{event.content}</span>
        </div>
      );

    case "action":
      return (
        <div className="box py-2 px-3 mb-1" style={{ borderLeft: "3px solid hsl(48, 100%, 67%)" }}>
          <span className="tag is-warning is-light is-small mr-1">
            {event.tool}
          </span>
          <span className="is-size-7 has-text-grey">
            {event.args ? JSON.stringify(event.args).slice(0, 200) : ""}
          </span>
        </div>
      );

    case "observation": {
      const content = event.content || "";
      const isSearch = event.tool === "search" || event.tool === "search_events";
      const isLook = event.tool === "look" || event.tool === "crop";

      if (isSearch) {
        const searchResults = parseSearchObservation(content);
        if (searchResults.length > 0) {
          return (
            <div className="box py-2 px-3 mb-1" style={{ borderLeft: "3px solid hsl(141, 53%, 53%)" }}>
              <div className="is-flex is-justify-content-space-between is-align-items-center">
                <span className="tag is-success is-light is-small mr-1">
                  {event.tool} ({searchResults.length} results)
                </span>
                <button
                  className="button is-ghost is-small p-0"
                  style={{ height: "auto", textDecoration: "none" }}
                  onClick={() => setExpanded(!expanded)}
                >
                  <span className="is-size-7 has-text-grey">{expanded ? "collapse" : "raw"}</span>
                </button>
              </div>
              <SearchResultStrip results={searchResults} onEnlargeImage={onEnlargeImage} />
              {expanded && (
                <pre className="is-size-7 mt-1" style={{ maxHeight: "120px", overflow: "auto", whiteSpace: "pre-wrap", background: "#f5f5f5", padding: "4px", borderRadius: "3px" }}>
                  {content}
                </pre>
              )}
            </div>
          );
        }
      }

      if (isLook) {
        const lookData = parseLookObservation(content);
        if (lookData) {
          return (
            <div className="box py-2 px-3 mb-1" style={{ borderLeft: "3px solid hsl(271, 76%, 53%)" }}>
              <span className="tag is-link is-light is-small mr-1 mb-1">
                {event.tool}
              </span>
              <LookResultDisplay data={lookData} onEnlargeImage={onEnlargeImage} />
            </div>
          );
        }
      }

      return (
        <div className="box py-2 px-3 mb-1" style={{ borderLeft: "3px solid hsl(141, 53%, 53%)" }}>
          <span className="tag is-success is-light is-small mr-1">
            {event.tool || "Observation"}
          </span>
          <pre
            className="is-size-7"
            style={{
              maxHeight: "120px",
              overflow: "auto",
              whiteSpace: "pre-wrap",
              background: "transparent",
              padding: 0,
            }}
          >
            {content.slice(0, 500)}
          </pre>
        </div>
      );
    }

    case "answer":
      return null;

    case "error":
      return (
        <div className="notification is-danger is-light py-2 px-3 mb-1">
          <span className="is-size-7">{event.message}</span>
        </div>
      );

    default:
      return null;
  }
}

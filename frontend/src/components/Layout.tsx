import { useCallback, useState } from "react";
import type {
  QueryItem,
  SearchConfig,
  SearchFilters,
  V1SearchRequest,
  V1SearchResponse,
  VQAResponse,
} from "../types";
import { useSearch } from "../hooks/useSearch";
import * as api from "../api/client";
import QueryPanel from "./QueryPanel";
import ResultsGrid from "./ResultsGrid";
import VideoModal from "./VideoModal";
import ImageLightbox from "./ImageLightbox";
import AnswerCard from "./AnswerCard";
import ChatTab from "./ChatTab";
import DashboardPage from "./DashboardPage";
import TracesPage from "./TracesPage";

type Tab = "search" | "chat" | "dashboard" | "traces";

export default function Layout() {
  const [tab, setTab] = useState<Tab>("search");
  const { results, loading, error, doSearch } = useSearch();
  const [video, setVideo] = useState<{
    videoName: string;
    ptsTime: number;
  } | null>(null);
  const [lightbox, setLightbox] = useState<string | null>(null);

  // v1 state
  const [v1Results, setV1Results] = useState<V1SearchResponse | null>(null);
  const [v1Loading, setV1Loading] = useState(false);
  const [v1Error, setV1Error] = useState<string | null>(null);
  const [upvoteIds, setUpvoteIds] = useState<string[]>([]);
  const [downvoteIds, setDownvoteIds] = useState<string[]>([]);

  // VLM ask state
  const [askQuestion, setAskQuestion] = useState("");
  const [askAnswer, setAskAnswer] = useState<VQAResponse | null>(null);
  const [askLoading, setAskLoading] = useState(false);
  const [askImageCount, setAskImageCount] = useState(3);

  const handleSearch = (
    queries: QueryItem[],
    config: SearchConfig,
    filters: SearchFilters,
    imageQuery?: string
  ) => {
    setV1Results(null);
    doSearch(queries, config, filters, imageQuery);
  };

  const handleV1Search = useCallback(async (req: V1SearchRequest) => {
    setV1Loading(true);
    setV1Error(null);
    setAskAnswer(null);
    setAskQuestion(req.parsed?.question || req.parsed?.events?.[0]?.text_en || "");
    try {
      const resp = await api.v1Search(req);
      setV1Results(resp);
    } catch (e) {
      setV1Error(e instanceof Error ? e.message : "Search failed");
    } finally {
      setV1Loading(false);
    }
  }, []);

  const handleAskVLM = useCallback(async () => {
    if (!askQuestion.trim() || askLoading || !v1Results?.results?.length) return;
    setAskLoading(true);
    setAskAnswer(null);
    try {
      const resp = await api.v1VQAAnswer({
        question: askQuestion.trim(),
        evidence_ids: v1Results.results.slice(0, askImageCount).map((result) => result.id),
        retrieval_trace_id: v1Results.trace_id,
      });
      setAskAnswer(resp);
    } catch {
      setAskAnswer({ status: "error", answer: "VLM request failed", evidence: [], trace_id: "", usage: {} });
    } finally {
      setAskLoading(false);
    }
  }, [askQuestion, askLoading, askImageCount, v1Results]);

  const toggleUpvote = useCallback((id: string) => {
    setUpvoteIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
    setDownvoteIds((prev) => prev.filter((x) => x !== id));
  }, []);

  const toggleDownvote = useCallback((id: string) => {
    setDownvoteIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
    setUpvoteIds((prev) => prev.filter((x) => x !== id));
  }, []);

  const isLoading = loading || v1Loading;
  const displayError = v1Error || error;

  const displayResults = v1Results
    ? {
        results: v1Results.results ?? [],
        translations: {} as Record<string, string>,
        search_time_ms: v1Results.latency_ms ?? 0,
      }
    : results;

  return (
    <>
      {/* Tab bar */}
      <div className="tabs is-boxed mb-0" style={{ paddingLeft: "1rem" }}>
        <ul>
          <li className={tab === "search" ? "is-active" : ""}>
            <a onClick={() => setTab("search")}>Search</a>
          </li>
          <li className={tab === "chat" ? "is-active" : ""}>
            <a onClick={() => setTab("chat")}>Chat</a>
          </li>
          <li className={tab === "dashboard" ? "is-active" : ""}>
            <a onClick={() => setTab("dashboard")}>Dashboard</a>
          </li>
          <li className={tab === "traces" ? "is-active" : ""}>
            <a onClick={() => setTab("traces")}>Traces</a>
          </li>
        </ul>
      </div>

      {tab === "search" && (
        <div className="columns is-gapless app-layout">
          <div className="column is-3 sidebar">
            <QueryPanel
              onSearch={handleSearch}
              onV1Search={handleV1Search}
              loading={isLoading}
              upvoteIds={upvoteIds}
              downvoteIds={downvoteIds}
            />
          </div>
          <div className="column results-area">
            {v1Results && (
              <div
                className="is-flex is-align-items-center px-3 pt-2"
                style={{ gap: "0.5rem" }}
              >
                <span className="tag is-info is-light is-small">
                  v1 | {(v1Results.results ?? []).length} results
                </span>
                <span className="tag is-light is-small">
                  {v1Results.latency_ms}ms
                </span>
                <span className="tag is-light is-small">
                  {v1Results.trace_id}
                </span>
              </div>
            )}
            {v1Results && (v1Results.results ?? []).length > 0 && (
              <div className="px-3 py-2">
                <div className="field has-addons mb-2">
                  <div className="control is-expanded">
                    <input
                      className="input is-small"
                      value={askQuestion}
                      onChange={(e) => setAskQuestion(e.target.value)}
                      placeholder="Ask about these results..."
                      onKeyDown={(e) => e.key === "Enter" && handleAskVLM()}
                    />
                  </div>
                  <div className="control">
                    <button
                      className="button is-small is-success"
                      onClick={handleAskVLM}
                      disabled={askLoading || !askQuestion.trim()}
                    >
                      {askLoading ? "Asking..." : "Ask VLM"}
                    </button>
                  </div>
                </div>
                <div className="field is-flex is-align-items-center mb-2" style={{ gap: "0.5rem" }}>
                  <label className="label is-small mb-0" htmlFor="ask-image-count">Top result images sent to VLM</label>
                  <input
                    id="ask-image-count"
                    className="input is-small"
                    type="number"
                    min={1}
                    max={12}
                    value={askImageCount}
                    onChange={(e) => setAskImageCount(Math.max(1, Math.min(12, Number(e.target.value) || 1)))}
                    style={{ width: 70 }}
                    disabled={askLoading}
                  />
                </div>
                {askAnswer && (
                  <AnswerCard
                    status={askAnswer.status}
                    answer={askAnswer.answer}
                    evidence={askAnswer.evidence}
                    traceId={askAnswer.trace_id}
                    onEnlargeImage={setLightbox}
                  />
                )}
              </div>
            )}
            <ResultsGrid
              response={displayResults}
              loading={isLoading}
              error={displayError}
              onClickKeyframe={(vn, pt) =>
                setVideo({ videoName: vn, ptsTime: pt })
              }
              onEnlargeImage={setLightbox}
              upvoteIds={upvoteIds}
              downvoteIds={downvoteIds}
              onUpvote={toggleUpvote}
              onDownvote={toggleDownvote}
            />
          </div>
        </div>
      )}

      <div style={{ display: tab === "chat" ? "block" : "none" }}><ChatTab /></div>
      {tab === "dashboard" && <DashboardPage />}
      {tab === "traces" && <TracesPage />}

      {video && (
        <VideoModal
          videoName={video.videoName}
          ptsTime={video.ptsTime}
          onClose={() => setVideo(null)}
        />
      )}
      {lightbox && (
        <ImageLightbox
          imageUrl={lightbox}
          onClose={() => setLightbox(null)}
        />
      )}
    </>
  );
}

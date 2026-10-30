import { useCallback, useMemo, useRef, useState } from "react";
import type { AgentEvent } from "../api/sse";
import { streamChat } from "../api/sse";
import AgentTimeline, {
  kfIdToImageUrl,
  parseSearchObservation,
  parseLookObservation,
} from "./AgentTimeline";
import AnswerCard from "./AnswerCard";
import ImageLightbox from "./ImageLightbox";

interface Turn {
  question: string;
  events: AgentEvent[];
  answer: AgentEvent | null;
}

export default function ChatTab() {
  const [question, setQuestion] = useState("");
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [answer, setAnswer] = useState<AgentEvent | null>(null);
  const [loading, setLoading] = useState(false);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [history, setHistory] = useState<Turn[]>([]);
  const [submittedQ, setSubmittedQ] = useState("");
  const [lightbox, setLightbox] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const handleSubmit = useCallback(async () => {
    const q = question.trim();
    if (!q || loading) return;

    // Push current turn to history if there was one
    if (events.length > 0 || answer) {
      setHistory((prev) => [
        ...prev,
        { question: submittedQ, events, answer },
      ]);
    }

    setLoading(true);
    setEvents([]);
    setAnswer(null);
    setSubmittedQ(q);
    setQuestion("");

    try {
      for await (const event of streamChat(q, 10, 120, threadId || undefined)) {
        if (event.type === "meta" && event.thread_id) {
          setThreadId(event.thread_id);
        } else if (event.type === "answer") {
          setAnswer(event);
        } else {
          setEvents((prev) => [...prev, event]);
        }
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
      }
    } catch (e) {
      setEvents((prev) => [
        ...prev,
        {
          type: "error",
          message: e instanceof Error ? e.message : "Stream failed",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }, [question, loading, threadId, events, answer, submittedQ]);

  const handleNewConversation = useCallback(() => {
    setThreadId(null);
    setHistory([]);
    setEvents([]);
    setAnswer(null);
    setQuestion("");
    setSubmittedQ("");
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      handleSubmit();
    }
  };

  // Collect all visual frames from current events
  const collectedFrames = useMemo(() => {
    const frames: { kfId: string; source: string; score?: number }[] = [];
    const seen = new Set<string>();
    const allEvents = [...history.flatMap((t) => t.events), ...events];
    for (const event of allEvents) {
      if (event.type === "observation") {
        if (event.tool === "search" || event.tool === "search_events") {
          const results = parseSearchObservation(event.content || "");
          for (const r of results) {
            if (!seen.has(r.kfId)) {
              seen.add(r.kfId);
              frames.push({ kfId: r.kfId, source: "search", score: r.score });
            }
          }
        } else if (event.tool === "look" || event.tool === "crop") {
          const data = parseLookObservation(event.content || "");
          if (data) {
            for (const kfId of data.frameIds) {
              if (!seen.has(kfId)) {
                seen.add(kfId);
                frames.push({ kfId, source: "look" });
              }
            }
          }
        }
      }
    }
    return frames;
  }, [history, events]);

  return (
    <>
      <div className="columns is-gapless" style={{ height: "calc(100vh - 52px)" }}>
        {/* Left: conversation */}
        <div className="column is-5" style={{ overflowY: "auto", padding: "1rem" }}>
          <div className="is-flex is-justify-content-space-between is-align-items-center mb-3">
            <h2 className="title is-5 mb-0">Chat (Agentic VQA)</h2>
            {threadId && (
              <button
                className="button is-small is-light"
                onClick={handleNewConversation}
              >
                New Conversation
              </button>
            )}
          </div>

          {/* Input */}
          <div className="field has-addons mb-3">
            <div className="control is-expanded">
              <input
                className="input"
                type="text"
                placeholder="Ask a question about the videos..."
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={loading}
              />
            </div>
            <div className="control">
              <button
                className={`button is-primary ${loading ? "is-loading" : ""}`}
                onClick={handleSubmit}
                disabled={loading || !question.trim()}
              >
                Ask
              </button>
            </div>
          </div>

          {/* Previous turns */}
          {history.map((turn, i) => (
            <div key={i} className="mb-3">
              <div className="is-size-7 has-text-weight-bold has-text-grey mb-1">
                Q: {turn.question}
              </div>
              <details className="mb-1">
                <summary className="is-size-7 has-text-grey" style={{ cursor: "pointer" }}>
                  {turn.events.length} events
                </summary>
                <AgentTimeline events={turn.events} onEnlargeImage={setLightbox} />
              </details>
              {turn.answer && (
                <AnswerCard
                  status={turn.answer.status || "unknown"}
                  answer={turn.answer.answer || ""}
                  evidence={turn.answer.evidence || []}
                  guardChanged={turn.answer.guard_changed}
                  onEnlargeImage={setLightbox}
                />
              )}
              <hr className="my-2" style={{ borderTop: "1px dashed #dbdbdb" }} />
            </div>
          ))}

          {/* Current turn */}
          <AgentTimeline events={events} onEnlargeImage={setLightbox} />

          {answer && (
            <AnswerCard
              status={answer.status || "unknown"}
              answer={answer.answer || ""}
              evidence={answer.evidence || []}
              guardChanged={answer.guard_changed}
              traceId={answer.trace_id}
              onEnlargeImage={setLightbox}
            />
          )}

          {loading && events.length > 0 && (
            <div className="has-text-centered py-2">
              <span className="is-size-7 has-text-grey">Agent is thinking...</span>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Right: visual results */}
        <div
          className="column is-7"
          style={{
            overflowY: "auto",
            padding: "1rem",
            borderLeft: "1px solid #eee",
            background: "#fafafa",
          }}
        >
          <h3 className="title is-6 mb-3">
            Visual Results ({collectedFrames.length})
          </h3>
          {collectedFrames.length === 0 && !loading && (
            <p className="has-text-grey is-size-7">
              Keyframes found by the agent will appear here.
            </p>
          )}
          <div
            className="is-flex is-flex-wrap-wrap"
            style={{ gap: "8px" }}
          >
            {collectedFrames.map((frame) => {
              const url = kfIdToImageUrl(frame.kfId);
              return (
                <div
                  key={frame.kfId}
                  style={{ width: "160px", cursor: "pointer" }}
                  onClick={() => setLightbox(url)}
                >
                  <figure className="image is-16by9" style={{ margin: 0 }}>
                    <img
                      src={url}
                      alt={frame.kfId}
                      loading="lazy"
                      style={{ objectFit: "cover", borderRadius: "4px" }}
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = "none";
                      }}
                    />
                  </figure>
                  <div className="is-size-7" style={{ lineHeight: 1.2, marginTop: "2px" }}>
                    <span className="has-text-weight-bold">{frame.kfId}</span>
                    {frame.score != null && (
                      <span className="ml-1 has-text-grey">
                        {frame.score.toFixed(3)}
                      </span>
                    )}
                    <span
                      className={`tag is-small ml-1 ${frame.source === "look" ? "is-link is-light" : "is-success is-light"}`}
                      style={{ fontSize: "0.6rem" }}
                    >
                      {frame.source}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {lightbox && (
        <ImageLightbox imageUrl={lightbox} onClose={() => setLightbox(null)} />
      )}
    </>
  );
}

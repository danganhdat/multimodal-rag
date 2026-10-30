/**
 * SSE client for /api/v1/chat streaming.
 * Uses fetch + ReadableStream to parse server-sent events.
 */

export interface AgentEvent {
  type: "thought" | "action" | "observation" | "answer" | "error" | "meta";
  content?: string;
  tool?: string;
  args?: Record<string, unknown>;
  status?: string;
  answer?: string;
  evidence?: string[];
  guard_changed?: boolean;
  message?: string;
  trace_id?: string;
  thread_id?: string;
}

export async function* streamChat(
  question: string,
  maxSteps = 10,
  timeoutS = 120,
  threadId?: string,
): AsyncGenerator<AgentEvent> {
  const res = await fetch("/api/v1/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      max_steps: maxSteps,
      timeout_s: timeoutS,
      thread_id: threadId || undefined,
    }),
  });

  if (!res.ok) {
    const text = await res.text();
    yield { type: "error", message: text };
    return;
  }

  const reader = res.body?.getReader();
  if (!reader) {
    yield { type: "error", message: "No response body" };
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith("data: ")) continue;

      const data = trimmed.slice(6);
      if (data === "[DONE]") return;

      try {
        yield JSON.parse(data) as AgentEvent;
      } catch {
        // skip malformed lines
      }
    }
  }
}

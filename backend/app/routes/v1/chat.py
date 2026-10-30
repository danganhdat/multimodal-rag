"""v1 chat endpoint: /api/v1/chat (agentic mode, SSE stream)."""

import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ...config_flags import load_flags
from ...observe import write_trace

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    max_steps: int = 10
    timeout_s: int = 120
    flags: dict | None = None
    thread_id: str | None = None


@router.post("/chat")
async def v1_chat(body: ChatRequest, request: Request):
    """Agentic chat: streams Thought/Action/Observation events via SSE.

    Each event is a JSON object on its own line, prefixed with "data: ".
    Event types: thought, action, observation, answer, error.
    """
    from ...agent.react import stream_agent
    from ...llm import LLMUnavailable

    trace_id = getattr(request.state, "trace_id", "")
    milvus = request.app.state.milvus
    siglip2 = request.app.state.siglip2_encoder
    bm25 = getattr(request.app.state, "bm25_index", None)
    flags = load_flags(body.flags or {})
    max_steps = body.flags.get("agent", {}).get("max_steps", body.max_steps) if body.flags else body.max_steps
    max_steps = max(1, min(30, int(max_steps)))
    logger.info("chat start [%s] max_steps=%d q=%.80s", trace_id, max_steps, body.question)

    async def event_stream():
        try:
            async for event in stream_agent(
                question=body.question,
                milvus=milvus,
                siglip2=siglip2,
                bm25=bm25,
                trace_id=trace_id,
                max_steps=max_steps,
                timeout_s=body.timeout_s,
                thread_id=body.thread_id,
                flags=flags,
            ):
                event["trace_id"] = trace_id
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        except LLMUnavailable as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'trace_id': trace_id})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)[:300], 'trace_id': trace_id})}\n\n"

        yield "data: [DONE]\n\n"

        write_trace(
            trace_id,
            endpoint="/api/v1/chat",
            mode="chat",
            question=body.question[:200],
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Trace-Id": trace_id,
        },
    )

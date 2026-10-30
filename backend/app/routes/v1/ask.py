"""v1 ask endpoint: /api/v1/ask (manual mode, selected frames only)."""

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ...config_flags import load_flags
from ...observe import span, write_trace

logger = logging.getLogger(__name__)
router = APIRouter()


class AskRequest(BaseModel):
    question: str
    kf_ids: list[str] = []
    max_steps: int = 10
    timeout_s: int = 120
    flags: dict | None = None


class AskResponse(BaseModel):
    status: str
    answer: str
    evidence: list[str] = []
    guard_changed: bool = False
    steps: int = 0
    trace_id: str = ""


@router.post("/ask", response_model=AskResponse)
async def v1_ask(body: AskRequest, request: Request):
    """Ask a question on user-selected keyframes (manual mode).

    The agent has look + neighbors + final_answer tools, but NOT search.
    """
    from ...agent.react import run_agent
    from ...llm import LLMUnavailable

    trace_id = getattr(request.state, "trace_id", "")
    milvus = request.app.state.milvus
    siglip2 = request.app.state.siglip2_encoder
    bm25 = getattr(request.app.state, "bm25_index", None)

    flags = load_flags(body.flags or {})
    max_steps = flags.get("agent", {}).get("max_steps", body.max_steps)
    logger.info("ask start [%s] max_steps=%d kf_ids=%d q=%.80s",
                trace_id, max_steps, len(body.kf_ids), body.question)

    if not body.question.strip():
        return AskResponse(
            status="clarify",
            answer="Please provide a question.",
            trace_id=trace_id,
        )

    try:
        result = await run_agent(
            question=body.question,
            milvus=milvus,
            siglip2=siglip2,
            bm25=bm25,
            mode="manual",
            candidates=body.kf_ids,
            trace_id=trace_id,
            max_steps=max_steps,
            timeout_s=body.timeout_s,
        )
    except LLMUnavailable as e:
        return AskResponse(
            status="error",
            answer=str(e),
            trace_id=trace_id,
        )

    write_trace(
        trace_id,
        endpoint="/api/v1/ask",
        mode="ask",
        question=body.question[:200],
        status=result.get("status", ""),
    )

    return AskResponse(
        status=result.get("status", "error"),
        answer=result.get("answer", ""),
        evidence=result.get("evidence", []),
        guard_changed=result.get("guard_changed", False),
        steps=result.get("steps", 0),
        trace_id=trace_id,
    )

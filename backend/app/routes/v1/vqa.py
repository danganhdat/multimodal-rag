"""v1 VQA endpoints: /api/v1/vqa and /api/v1/vqa/answer."""

import time

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ...config_flags import load_flags, flags_id
from ...observe import span, write_trace

router = APIRouter()


class VQAAnswerRequest(BaseModel):
    """One-shot MLLM answer from user-selected evidence."""
    question: str
    evidence_ids: list[str]
    retrieval_trace_id: str = ""
    flags: dict | None = None


class VQARequest(BaseModel):
    """End-to-end one-shot: parse → retrieve → evidence → answer."""
    query: str
    k: int = 200
    flags: dict | None = None


class VQAResponse(BaseModel):
    status: str
    answer: str = ""
    evidence: list[str] = []
    trace_id: str = ""
    search_time_ms: float = 0.0
    answer_time_ms: float = 0.0
    usage: dict = {}


@router.post("/vqa/answer", response_model=VQAResponse)
async def v1_vqa_answer(body: VQAAnswerRequest, request: Request):
    """Answer a question using user-selected evidence frames (Manual Mode).

    Does NOT run retrieval — uses the provided evidence_ids directly.
    Does NOT enter a ReAct loop — one-shot MLLM only.
    """
    trace_id = getattr(request.state, "trace_id", "")
    milvus = request.app.state.milvus

    with span(trace_id, "vqa.answer", question=body.question[:200],
              n_evidence=len(body.evidence_ids)):
        t0 = time.perf_counter()

        from ...evidence import build_evidence_from_ids
        evidence = build_evidence_from_ids(
            kf_ids=body.evidence_ids,
            milvus=milvus,
            trace_id=trace_id,
        )

        from ...vqa import generate_answer
        result = await generate_answer(
            question=body.question,
            evidence=evidence,
            trace_id=trace_id,
        )

        answer_ms = (time.perf_counter() - t0) * 1000

    write_trace(trace_id, endpoint="/api/v1/vqa/answer", mode="vqa_answer",
                question=body.question[:200], status=result["status"],
                workflow_id=body.retrieval_trace_id)

    return VQAResponse(
        status=result["status"],
        answer=result["answer"],
        evidence=result["evidence"],
        trace_id=trace_id,
        answer_time_ms=round(answer_ms, 2),
        usage=result.get("usage", {}),
    )


@router.post("/vqa", response_model=VQAResponse)
async def v1_vqa(body: VQARequest, request: Request):
    """End-to-end one-shot VQA: parse → retrieve → evidence → answer.

    NOT a ReAct agent — runs the full pipeline once and returns.
    """
    trace_id = getattr(request.state, "trace_id", "")
    siglip2 = request.app.state.siglip2_encoder
    milvus = request.app.state.milvus
    bm25 = getattr(request.app.state, "bm25_index", None)

    flags = load_flags(body.flags)
    cid = flags_id(flags)
    vqa_flags = flags.get("vqa", {})
    top_k_evidence = max(1, min(12, int(vqa_flags.get("top_k_evidence", 3))))

    with span(trace_id, "vqa.e2e", question=body.query[:200]):
        # Step 1: Parse query
        t0 = time.perf_counter()
        from ...query.qe import parse_query
        parsed = await parse_query(body.query, flags=flags, trace_id=trace_id)
        enabled_events = [e for e in parsed.events if e.enabled]

        if not enabled_events:
            return VQAResponse(
                status="error",
                answer="Could not parse query into events.",
                trace_id=trace_id,
            )

        # Step 2: Search
        query_vectors = []
        events_data = []
        for event in enabled_events:
            text = event.text_en.strip() or event.text_vi.strip()
            if text:
                query_vectors.append(siglip2.encode_text(text))
            else:
                query_vectors.append(siglip2.encode_text("keyframe"))
            events_data.append({
                "id": event.id,
                "text_en": event.text_en,
                "text_vi": event.text_vi,
                "ocr": event.ocr,
                "objects": event.objects,
                "variants": event.variants,
            })

        answer_event = min(parsed.answer_event, len(enabled_events) - 1)

        # Thread requires_temporal
        search_flags = dict(flags)
        temporal = dict(search_flags.get("temporal", {}))
        temporal["requires_temporal"] = parsed.requires_temporal
        search_flags["temporal"] = temporal

        from ...pipeline import run_search_multi_event
        results = await run_search_multi_event(
            milvus=milvus,
            siglip2_encoder=siglip2,
            bm25_index=bm25,
            events=events_data,
            query_vectors=query_vectors,
            answer_event=answer_event,
            flags=search_flags,
            trace_id=trace_id,
        )

        search_ms = (time.perf_counter() - t0) * 1000

        if not results:
            return VQAResponse(
                status="insufficient_evidence",
                answer="KHONG_TRA_LOI_DUOC",
                trace_id=trace_id,
                search_time_ms=round(search_ms, 2),
            )

        # Step 3: Build evidence
        from ...evidence import build_evidence
        evidence = build_evidence(
            results=results,
            events=events_data if len(events_data) >= 2 else None,
            top_k=top_k_evidence,
            trace_id=trace_id,
        )

        # Step 4: Generate answer
        t1 = time.perf_counter()
        from ...vqa import generate_answer
        result = await generate_answer(
            question=body.query,
            evidence=evidence,
            trace_id=trace_id,
        )
        answer_ms = (time.perf_counter() - t1) * 1000

    write_trace(trace_id, endpoint="/api/v1/vqa", mode="vqa",
                question=body.query[:200], status=result["status"],
                config_id=cid)

    return VQAResponse(
        status=result["status"],
        answer=result["answer"],
        evidence=result["evidence"],
        trace_id=trace_id,
        search_time_ms=round(search_ms, 2),
        answer_time_ms=round(answer_ms, 2),
        usage=result.get("usage", {}),
    )

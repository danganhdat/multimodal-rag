"""v1 trace endpoints: /api/v1/traces, /api/v1/traces/{id}, /api/v1/traces/{id}/sequence."""

import json

from fastapi import APIRouter, Query

from ...observe import get_traces, get_trace

router = APIRouter()


@router.get("/traces")
async def v1_traces(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    return {"traces": get_traces(limit=limit, offset=offset)}


@router.get("/traces/{trace_id}")
async def v1_trace_detail(trace_id: str):
    data = get_trace(trace_id)
    if data is None:
        return {"error": {"code": 404, "message": "Trace not found"}}
    return data


@router.get("/traces/{trace_id}/sequence")
async def v1_trace_sequence(trace_id: str):
    """Generate a Mermaid sequenceDiagram from spans."""
    data = get_trace(trace_id)
    if data is None:
        return {"error": {"code": 404, "message": "Trace not found"}}

    spans = data.get("spans", [])
    if not spans:
        return {"mermaid": "sequenceDiagram\n  Note over API: No spans recorded"}

    # Map span names to participants
    participant_map = {
        "request": "API",
        "qe.parse": "LLM",
        "qe.fallback": "API",
        "search.encode_text": "API",
        "search.encode_events": "API",
        "pipeline.dense": "Milvus",
        "search.dense": "Milvus",
        "pipeline.bm25": "BM25",
        "search.bm25": "BM25",
        "search.upvote": "Milvus",
        "search.image_encode": "API",
        "fusion.rrf": "API",
        "metadata.boost": "API",
        "temporal.search": "Temporal",
        "temporal": "Temporal",
        "rerank.superglobal": "Rerank",
        "rerank.mllm": "VLM",
        "rerank.vlm": "VLM",
        "evidence.build": "API",
        "evidence.build_from_ids": "API",
        "vqa.generate": "VLM",
        "vqa.answer": "VLM",
        "vqa.e2e": "API",
        "agent.run": "Agent",
        "agent.step": "Agent",
        "agent.stream": "Agent",
        "agent.guard": "Agent",
        "tool.search": "Milvus",
        "tool.search_events": "Milvus",
        "tool.look": "VLM",
        "tool.crop": "VLM",
        "tool.read_ocr": "Milvus",
        "tool.neighbors": "Milvus",
    }

    participants_seen = set()
    lines = ["sequenceDiagram"]

    for s in spans:
        name = s.get("name", "")
        ms = s.get("latency_ms", 0)
        participant = participant_map.get(name, "API")

        if participant not in participants_seen:
            participants_seen.add(participant)
            lines.append(f"  participant {participant}")

    # Add arrows
    for s in spans:
        name = s.get("name", "")
        ms = s.get("latency_ms", 0)
        # Match dynamic span names like retrieve.dense.e0, fusion.rrf.e1
        base_name = name.rsplit(".e", 1)[0] if ".e" in name and name[-1].isdigit() else name
        participant = participant_map.get(name, participant_map.get(base_name, "API"))
        attrs = json.loads(s.get("attrs", "{}"))
        err = s.get("error", "")

        arrow = "->>" if not err else "-x"
        label = f"{name} ({ms:.0f}ms)"
        if err:
            label += " ERROR"

        if participant != "API":
            lines.append(f"  API{arrow}{participant}: {label}")
            lines.append(f"  {participant}-->>API: done")
        else:
            lines.append(f"  Note over API: {label}")

    return {"mermaid": "\n".join(lines)}

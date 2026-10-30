"""v1 search endpoints: /api/v1/search, /api/v1/search/image, /api/v1/search/sketch."""

import asyncio
import base64
import io
import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

from ...config import settings
from ...config_flags import load_flags, flags_id
from ...observe import span, write_trace
from ...query.qe import EventQuery, ParsedQuery

router = APIRouter()


# ── Request / Response models ──

class SearchQuery(BaseModel):
    """A single text query for v1 search."""
    text: str
    lang: str = "en"
    ocr: list[str] = []


class V1SearchRequest(BaseModel):
    """POST /api/v1/search"""
    queries: list[SearchQuery] = []
    text: str | None = None
    parsed: ParsedQuery | None = None
    k: int = 200
    upvote_ids: list[str] = []
    downvote_ids: list[str] = []
    upvote_kf_ids: list[str] = []
    downvote_kf_ids: list[str] = []
    flags: dict | None = None


class ImageSearchRequest(BaseModel):
    """POST /api/v1/search/image"""
    kf_id: str | None = None
    image_b64: str | None = None
    k: int = 200
    flags: dict | None = None


class NeighborFrame(BaseModel):
    keyframe_idx: int = 0
    frame_idx: int = 0
    pts_time: float = 0.0
    image_url: str = ""


class V1SearchResult(BaseModel):
    id: str
    video_name: str = ""
    keyframe_idx: int = 0
    frame_idx: int = 0
    pts_time: float = 0.0
    fps: float = 0.0
    score: float = 0.0
    sources: dict = {}
    objects: list[str] = []
    ocr: str = ""
    image_url: str = ""
    neighbors: list[NeighborFrame] = []


class V1SearchResponse(BaseModel):
    results: list[V1SearchResult]
    translations: dict[str, str] = {}
    latency_ms: float = 0.0
    trace_id: str = ""
    config_id: str = ""


# ── Helpers ──

async def _translate_if_needed(text: str, lang: str) -> tuple[str, dict[str, str]]:
    """Translate vi→en if needed. Returns (english_text, {vi: en})."""
    translations = {}
    if lang == "vi":
        try:
            from ...translator import translate_vi_to_en
            translated = await asyncio.to_thread(translate_vi_to_en, text)
            translations[text] = translated
            return translated, translations
        except Exception:
            translations[text] = text
    return text, translations


def _build_filter_expr(filters: dict | None) -> str | None:
    if not filters:
        return None
    parts = []
    if filters.get("objects"):
        normalized = [o.lower() for o in filters["objects"]]
        parts.append(f"ARRAY_CONTAINS_ANY(objects, {json.dumps(normalized)})")
    for term in filters.get("ocr", []):
        safe = term.replace('"', '\\"')
        parts.append(f'ocr like "%{safe}%"')
    return " AND ".join(parts) if parts else None


def _format_results(raw: list[dict], milvus=None, neighbor_window: int = 5) -> list[V1SearchResult]:
    base = settings.IMAGE_BASE_URL
    results = []
    for r in raw:
        vname = r.get("video_name", "")
        kidx = r.get("keyframe_idx", 0)
        results.append(V1SearchResult(
            id=r["id"],
            video_name=vname,
            keyframe_idx=kidx,
            frame_idx=r.get("frame_idx", 0),
            pts_time=r.get("pts_time", 0.0),
            fps=r.get("fps", 0.0),
            score=r.get("rrf_score", r.get("score", 0.0)),
            sources=r.get("sources", {}),
            objects=r.get("objects", []),
            ocr=r.get("ocr", ""),
            image_url=f"{base}/{vname}/{kidx:03d}.jpg" if vname else "",
        ))

    if milvus and results:
        items = [(r.video_name, r.keyframe_idx) for r in results]
        batch = milvus.get_neighbors_batch(items, window=neighbor_window)
        for r in results:
            key = f"{r.video_name}/{r.keyframe_idx:03d}"
            raw_neighbors = batch.get(key, [])
            r.neighbors = [
                NeighborFrame(
                    keyframe_idx=n["keyframe_idx"],
                    frame_idx=n.get("frame_idx", 0),
                    pts_time=n.get("pts_time", 0.0),
                    image_url=f"{base}/{r.video_name}/{n['keyframe_idx']:03d}.jpg",
                )
                for n in raw_neighbors
            ]

    return results


# ── Endpoints ──

@router.post("/search", response_model=V1SearchResponse)
async def v1_search(body: V1SearchRequest, request: Request):
    """Search with text queries or a ParsedQuery, optional upvote/downvote."""
    import time

    trace_id = getattr(request.state, "trace_id", "")
    siglip2 = request.app.state.siglip2_encoder
    milvus = request.app.state.milvus
    bm25 = getattr(request.app.state, "bm25_index", None)

    flags = load_flags(body.flags)
    cid = flags_id(flags)

    # Merge old and new upvote/downvote field names
    up_ids = body.upvote_ids or body.upvote_kf_ids or []
    down_ids = body.downvote_ids or body.downvote_kf_ids or []

    t0 = time.perf_counter()

    # Route A: ParsedQuery with events (from QE)
    if body.parsed and body.parsed.events:
        raw_results = await _search_parsed(
            body.parsed, siglip2, milvus, bm25,
            up_ids, down_ids, flags, trace_id,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        results = _format_results(raw_results[:body.k], milvus=milvus)
        write_trace(trace_id, mode="search_parsed",
                    n_results=len(results), config_id=cid)
        return V1SearchResponse(
            results=results,
            latency_ms=round(elapsed, 2),
            trace_id=trace_id,
            config_id=cid,
        )

    # Route B: Simple text queries (legacy or shorthand)
    from ...pipeline import run_search

    queries = list(body.queries)
    if body.text and not queries:
        queries = [SearchQuery(text=body.text)]

    if not queries and not up_ids:
        return V1SearchResponse(results=[], trace_id=trace_id, config_id=cid)

    translations: dict[str, str] = {}
    text_vectors: list[list[float]] = []
    ocr_queries: list[str] = []
    first_en_text = ""

    with span(trace_id, "search.encode_text", n_queries=len(queries)):
        for q in queries:
            text = q.text.strip()
            if not text:
                continue
            en_text, trans = await _translate_if_needed(text, q.lang)
            translations.update(trans)
            text_vectors.append(siglip2.encode_text(en_text))
            ocr_queries.extend(q.ocr)
            if not first_en_text:
                first_en_text = en_text

    raw_results = await run_search(
        milvus=milvus,
        siglip2_encoder=siglip2,
        bm25_index=bm25,
        text_vectors=text_vectors,
        ocr_queries=ocr_queries if ocr_queries else None,
        upvote_ids=up_ids or None,
        downvote_ids=down_ids or None,
        flags=flags,
        query_text=first_en_text,
        trace_id=trace_id,
    )

    elapsed = (time.perf_counter() - t0) * 1000
    results = _format_results(raw_results[:body.k], milvus=milvus)

    write_trace(trace_id, mode="search", n_results=len(results),
                config_id=cid, encoder="siglip2",
                index_version=settings.INDEX_VERSION)

    return V1SearchResponse(
        results=results,
        translations=translations,
        latency_ms=round(elapsed, 2),
        trace_id=trace_id,
        config_id=cid,
    )


async def _search_parsed(parsed, siglip2, milvus, bm25,
                          up_ids, down_ids, flags, trace_id):
    """Run search from a ParsedQuery with events."""
    enabled_events = [e for e in parsed.events if e.enabled]
    if not enabled_events:
        return []

    # Encode each event's text_en (primary vector only — variants encoded in pipeline)
    query_vectors = []
    events_data = []
    with span(trace_id, "search.encode_events", n_events=len(enabled_events)):
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

    # Determine answer_event index within enabled events
    answer_event = min(parsed.answer_event, len(enabled_events) - 1)

    # Thread requires_temporal into temporal flags
    search_flags = dict(flags)
    temporal = dict(search_flags.get("temporal", {}))
    temporal["requires_temporal"] = getattr(parsed, "requires_temporal", False)
    search_flags["temporal"] = temporal

    if len(enabled_events) >= 2:
        from ...pipeline import run_search_multi_event
        return await run_search_multi_event(
            milvus=milvus,
            siglip2_encoder=siglip2,
            bm25_index=bm25,
            events=events_data,
            query_vectors=query_vectors,
            answer_event=answer_event,
            upvote_ids=up_ids or None,
            downvote_ids=down_ids or None,
            flags=search_flags,
            trace_id=trace_id,
        )
    else:
        from ...pipeline import run_search
        return await run_search(
            milvus=milvus,
            siglip2_encoder=siglip2,
            bm25_index=bm25,
            text_vectors=query_vectors,
            ocr_queries=events_data[0].get("ocr") or None,
            upvote_ids=up_ids or None,
            downvote_ids=down_ids or None,
            flags=search_flags,
            query_text=events_data[0].get("text_en", ""),
            trace_id=trace_id,
        )


@router.post("/search/image", response_model=V1SearchResponse)
async def v1_search_image(body: ImageSearchRequest, request: Request):
    """Image-by-example search: pass a kf_id or an uploaded image."""
    import time

    trace_id = getattr(request.state, "trace_id", "")
    siglip2 = request.app.state.siglip2_encoder
    milvus = request.app.state.milvus

    flags = load_flags(body.flags)
    cid = flags_id(flags)

    t0 = time.perf_counter()
    text_vectors: list[list[float]] = []

    with span(trace_id, "search.image_encode") as attrs:
        if body.kf_id:
            # Use the existing text_vector from Milvus
            vecs = milvus.get_vectors([body.kf_id], "text_vector")
            if body.kf_id in vecs:
                text_vectors.append(vecs[body.kf_id])
            attrs["source"] = "kf_id"
            attrs["kf_id"] = body.kf_id
        elif body.image_b64:
            # Encode uploaded image with SigLIP2
            img_bytes = base64.b64decode(body.image_b64)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            text_vectors.append(siglip2.encode_image(image))
            attrs["source"] = "upload"

    if not text_vectors:
        return V1SearchResponse(results=[], trace_id=trace_id, config_id=cid)

    from ...pipeline import run_search

    raw_results = await run_search(
        milvus=milvus,
        siglip2_encoder=siglip2,
        bm25_index=None,
        text_vectors=text_vectors,
        flags=flags,
        trace_id=trace_id,
    )

    elapsed = (time.perf_counter() - t0) * 1000
    results = _format_results(raw_results[:body.k], milvus=milvus)

    write_trace(trace_id, mode="image_search", n_results=len(results), config_id=cid)

    return V1SearchResponse(
        results=results,
        latency_ms=round(elapsed, 2),
        trace_id=trace_id,
        config_id=cid,
    )


@router.get("/config")
async def v1_config(request: Request):
    """Return current flags and system status."""
    flags = load_flags()
    bm25 = getattr(request.app.state, "bm25_index", None)
    llm_key = bool(settings.LLM_API_KEY)

    return {
        "flags": flags,
        "config_id": flags_id(flags),
        "status": {
            "bm25": bm25.ready if bm25 else False,
            "bm25_n_docs": bm25.n_docs if bm25 and bm25.ready else 0,
            "llm": llm_key and bool(settings.LLM_MODEL),
            "vlm": llm_key and bool(settings.VLM_MODEL),
            "milvus": True,
            "llm_model": settings.LLM_MODEL if llm_key else None,
            "vlm_model": settings.VLM_MODEL if llm_key else None,
            "index_version": settings.INDEX_VERSION,
            "encoder": "siglip2",
            "collection": settings.COLLECTION_NAME,
        },
    }

import asyncio
import base64
import io
import json
import time
from collections import defaultdict

from fastapi import APIRouter, Request
from PIL import Image

from ..config import settings
from ..models.schemas import (
    NeighborFrame,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

router = APIRouter()


def _build_filter_expr(filters) -> str | None:
    if not filters:
        return None
    parts = []
    if filters.objects:
        normalized = [o.lower() for o in filters.objects]
        parts.append(f"ARRAY_CONTAINS_ANY(objects, {json.dumps(normalized)})")
    for term in filters.ocr:
        safe = term.replace('"', '\\"')
        parts.append(f'ocr like "%{safe}%"')
    return " AND ".join(parts) if parts else None


async def _encode_queries(queries, siglip2_encoder):
    """Encode text queries using SigLIP2 (the only text encoder now)."""
    translations: dict[str, str] = {}
    text_vectors: list[list[float]] = []

    for q in queries:
        text = q.text.strip()
        if not text:
            continue

        if q.lang == "vi":
            try:
                from ..translator import translate_vi_to_en
                translated = await asyncio.to_thread(translate_vi_to_en, text)
                translations[text] = translated
                text = translated
            except Exception:
                translations[text] = text

        text_vectors.append(siglip2_encoder.encode_text(text))

    return text_vectors, translations


def _client_merge(vectors, milvus, cfg, filter_expr, anns_field="text_vector"):
    """Client-side RRF merge of multiple vector queries."""
    score_map: dict[str, dict] = {}
    rank_map: defaultdict[str, list[int]] = defaultdict(list)

    for vec in vectors:
        hits = milvus.search_single(
            vector=vec,
            top_k=cfg.sub_k,
            metric_type=cfg.metric_type.value,
            filter_expr=filter_expr,
            anns_field=anns_field,
        )
        for rank, h in enumerate(hits):
            hid = h["id"]
            rank_map[hid].append(rank)
            if hid not in score_map:
                score_map[hid] = h

    merged = []
    for hid, ranks in rank_map.items():
        rrf_score = sum(1.0 / (cfg.rrf_k + r + 1) for r in ranks)
        entry = dict(score_map[hid])
        entry["score"] = rrf_score
        merged.append(entry)
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged


def _build_results(raw_hits, milvus, cfg) -> list[SearchResult]:
    results = []
    base = settings.IMAGE_BASE_URL

    for hit in raw_hits:
        vname = hit["video_name"]
        neighbors_raw = milvus.get_neighbors(
            video_name=vname,
            keyframe_idx=hit["keyframe_idx"],
            window=cfg.neighbor_window,
        )
        neighbors = [
            NeighborFrame(
                keyframe_idx=n["keyframe_idx"],
                frame_idx=n["frame_idx"],
                pts_time=n["pts_time"],
                image_url=f"{base}/{vname}/{n['keyframe_idx']:03d}.jpg",
            )
            for n in neighbors_raw
            if n["keyframe_idx"] != hit["keyframe_idx"]
        ]
        results.append(
            SearchResult(
                id=hit["id"],
                video_name=vname,
                keyframe_idx=hit["keyframe_idx"],
                frame_idx=hit["frame_idx"],
                pts_time=hit["pts_time"],
                fps=hit["fps"],
                score=hit["score"],
                objects=hit.get("objects", []),
                ocr=hit.get("ocr", ""),
                neighbors=neighbors,
                image_url=f"{base}/{vname}/{hit['keyframe_idx']:03d}.jpg",
            )
        )
    return results


@router.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, request: Request):
    siglip2_encoder = request.app.state.siglip2_encoder
    sketch_encoder = getattr(request.app.state, "sketch_encoder", None)
    milvus = request.app.state.milvus
    cfg = body.config

    text_vectors, translations = await _encode_queries(body.queries, siglip2_encoder)

    # Sketch goes through clip_vector via SketchEncoder
    clip_vectors: list[list[float]] = []
    if body.image_query and sketch_encoder:
        img_bytes = base64.b64decode(body.image_query)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        clip_vectors.append(sketch_encoder.encode_image(image))

    if not text_vectors and not clip_vectors:
        return SearchResponse(results=[], search_time_ms=0)

    filter_expr = _build_filter_expr(body.filters)
    retrieval_k = cfg.top_k

    t0 = time.perf_counter()

    # Determine search strategy
    if clip_vectors and text_vectors:
        # Both sketch + text: merge via client-side RRF
        all_vectors = text_vectors + clip_vectors
        anns_fields = ["text_vector"] * len(text_vectors) + ["clip_vector"] * len(clip_vectors)
        raw_hits = milvus.search_hybrid(
            vectors=all_vectors,
            top_k=retrieval_k,
            sub_k=cfg.sub_k,
            metric_type=cfg.metric_type.value,
            ranker_type=cfg.ranker.value,
            rrf_k=cfg.rrf_k,
            weights=cfg.weights if cfg.weights else None,
            filter_expr=filter_expr,
            anns_fields=anns_fields,
        )
    elif len(text_vectors) == 1 and not clip_vectors:
        # Single text query: ANN
        raw_hits = milvus.search_single(
            vector=text_vectors[0],
            top_k=retrieval_k,
            metric_type=cfg.metric_type.value,
            filter_expr=filter_expr,
            group_by_video=cfg.group_by_video,
            anns_field="text_vector",
        )
    elif len(clip_vectors) == 1 and not text_vectors:
        # Sketch only: ANN on clip_vector
        raw_hits = milvus.search_single(
            vector=clip_vectors[0],
            top_k=retrieval_k,
            metric_type=cfg.metric_type.value,
            filter_expr=filter_expr,
            group_by_video=cfg.group_by_video,
            anns_field="clip_vector",
        )
    elif cfg.use_client_merge:
        raw_hits = _client_merge(text_vectors, milvus, cfg, filter_expr, "text_vector")[:retrieval_k]
    else:
        raw_hits = milvus.search_hybrid(
            vectors=text_vectors,
            top_k=retrieval_k,
            sub_k=cfg.sub_k,
            metric_type=cfg.metric_type.value,
            ranker_type=cfg.ranker.value,
            rrf_k=cfg.rrf_k,
            weights=cfg.weights if cfg.weights else None,
            filter_expr=filter_expr,
            anns_fields=["text_vector"] * len(text_vectors),
        )

    elapsed = (time.perf_counter() - t0) * 1000
    results = _build_results(raw_hits, milvus, cfg)

    return SearchResponse(
        results=results,
        translations=translations,
        search_time_ms=round(elapsed, 2),
    )

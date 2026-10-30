"""Search pipeline: runs stages S2→S5 according to feature flags.

Each stage is a separate span for observability. The pipeline takes
encoded vectors and returns ranked results with sources.
"""

import asyncio

from .config import settings
from .observe import span


def _fuse_rerank_rrf(
    sg_results: list[dict],
    mllm_results: list[dict],
    k: int = 10,
) -> list[dict]:
    """RRF fusion of SuperGlobal and MLLM reranker outputs."""
    score_map: dict[str, float] = {}
    result_map: dict[str, dict] = {}

    for rank, r in enumerate(sg_results):
        rid = r["id"]
        score_map[rid] = score_map.get(rid, 0) + 1.0 / (k + rank + 1)
        result_map[rid] = r

    for rank, r in enumerate(mllm_results):
        rid = r["id"]
        score_map[rid] = score_map.get(rid, 0) + 1.0 / (k + rank + 1)
        if rid not in result_map:
            result_map[rid] = r
        else:
            if "mllm_rank" in r:
                result_map[rid]["mllm_rank"] = r["mllm_rank"]
            result_map[rid].setdefault("sources", {}).update(
                r.get("sources", {})
            )

    ranked_ids = sorted(score_map, key=lambda x: score_map[x], reverse=True)
    return [result_map[rid] for rid in ranked_ids]


def _encode_variants(siglip2, text_en: str, variants: list[str]) -> list[list[float]]:
    """Encode text_en + variants into vectors for multi-query RRF."""
    texts = [text_en] + [v for v in variants if v.strip()]
    return [siglip2.encode_text(t) for t in texts]


async def run_search(
    milvus,
    siglip2_encoder,
    bm25_index,
    text_vectors: list[list[float]],
    ocr_queries: list[str] | None = None,
    upvote_ids: list[str] | None = None,
    downvote_ids: list[str] | None = None,
    flags: dict | None = None,
    filter_expr: str | None = None,
    query_text: str = "",
    trace_id: str = "",
) -> list[dict]:
    """Run the search pipeline: dense + BM25 + upvote → RRF → results.

    Args:
        milvus: MilvusSearchService instance.
        siglip2_encoder: SigLIP2Encoder for image-by-example.
        bm25_index: BM25Index (may be None/not ready).
        text_vectors: Encoded text query vectors (SigLIP2).
        ocr_queries: OCR text strings for BM25 (from QE or user).
        upvote_ids: kf_ids the user upvoted (image-by-example signal).
        downvote_ids: kf_ids the user downvoted.
        flags: Feature flags dict (from config_flags.load_flags).
        filter_expr: Milvus filter expression.
        trace_id: For observability.

    Returns:
        List of result dicts with: id, rrf_score, sources, video_name,
        keyframe_idx, frame_idx, pts_time, fps, objects, ocr.
    """
    from .retrieve.dense import search_dense
    from .retrieve.bm25 import search_bm25
    from .retrieve.fusion import rrf_fuse, apply_downvote_penalty

    if flags is None:
        from .config_flags import get_defaults
        flags = get_defaults()

    search_flags = flags.get("search", {})
    manual_flags = flags.get("manual", {})
    rerank_flags = flags.get("rerank", {})
    top_n = search_flags.get("top_n", 200)
    rrf_k = search_flags.get("rrf_k", 60)

    rrf_lists: list[tuple[str, list]] = []

    # S2a: Dense ANN search
    if search_flags.get("dense", True) and text_vectors:
        with span(trace_id, "retrieve.dense", n_vectors=len(text_vectors)):
            dense_results = search_dense(
                milvus, text_vectors, top_n=top_n,
                filter_expr=filter_expr, trace_id=trace_id,
            )
            for i, hits in enumerate(dense_results):
                rrf_lists.append((f"dense_{i}", hits))

    # S2b: BM25 lexical search (disabled by default)
    if (search_flags.get("bm25", False)
            and bm25_index is not None
            and bm25_index.ready
            and ocr_queries):
        with span(trace_id, "retrieve.bm25"):
            bm25_results = search_bm25(
                bm25_index, ocr_queries, top_n=top_n, trace_id=trace_id,
            )
            for i, hits in enumerate(bm25_results):
                if hits:
                    rrf_lists.append((f"bm25_{i}", hits))

    # Upvote: image-by-example signal
    if upvote_ids:
        with span(trace_id, "search.upvote", n_upvotes=len(upvote_ids)) as attrs:
            upvote_vecs = milvus.get_vectors(upvote_ids, "text_vector")
            for uid, vec in upvote_vecs.items():
                up_hits = milvus.search_batch(
                    vectors=[vec],
                    anns_field="text_vector",
                    top_k=top_n,
                    filter_expr=filter_expr,
                )
                if up_hits and up_hits[0]:
                    rrf_lists.append((f"upvote_{uid}", up_hits[0]))
            attrs["n_vectors"] = len(upvote_vecs)

    if not rrf_lists:
        return []

    # Fuse via RRF
    results = rrf_fuse(rrf_lists, k=rrf_k, top_n=top_n, trace_id=trace_id)

    # Fill in metadata for items that only came from BM25/upvote (no metadata yet)
    _backfill_metadata(results, milvus)

    # S3: Metadata soft boost (after RRF, before rerank)
    boost_flags = flags.get("metadata_boost", {})
    if boost_flags.get("enabled", False) and query_text:
        from .retrieve.metadata_boost import apply_metadata_boost
        results = apply_metadata_boost(
            results,
            query_terms=query_text.lower().split(),
            alpha=boost_flags.get("alpha", 0.05),
            max_boost=boost_flags.get("max_boost", 0.1),
            trace_id=trace_id,
        )

    # Downvote penalty
    if downvote_ids:
        dv_vecs_map = milvus.get_vectors(downvote_ids, "text_vector")
        dv_vecs = list(dv_vecs_map.values())
        if dv_vecs:
            penalty = manual_flags.get("downvote_penalty", 0.5)
            results = apply_downvote_penalty(results, dv_vecs, milvus, penalty=penalty)

    # S5: Rerank
    rerank_method = rerank_flags.get("method", "superglobal")
    if rerank_method == "sg+mllm" and text_vectors:
        from .rerank.superglobal import rerank_superglobal
        from .rerank.mllm import rerank_mllm

        with span(trace_id, "rerank.sg_mllm") as attrs:
            sg_task = asyncio.to_thread(
                rerank_superglobal,
                list(results),
                query_vec=text_vectors[0],
                milvus=milvus,
                top=rerank_flags.get("top", 50),
                m=rerank_flags.get("sg_m", 3),
                trace_id=trace_id,
            )
            if query_text:
                mllm_task = rerank_mllm(
                    list(results),
                    query=query_text,
                    top=rerank_flags.get("mllm_top", 10),
                    timeout_ms=rerank_flags.get("timeout_ms", 5000),
                    fallback_to_previous=True,
                    trace_id=trace_id,
                )
                sg_out, mllm_out = await asyncio.gather(sg_task, mllm_task)
                results = _fuse_rerank_rrf(sg_out, mllm_out)
                attrs["fusion"] = "sg+mllm"
            else:
                results = await sg_task
                attrs["fusion"] = "sg_only"

    elif rerank_method == "superglobal" and text_vectors:
        from .rerank.superglobal import rerank_superglobal
        results = rerank_superglobal(
            results,
            query_vec=text_vectors[0],
            milvus=milvus,
            top=rerank_flags.get("top", 50),
            m=rerank_flags.get("sg_m", 3),
            trace_id=trace_id,
        )
    elif rerank_method == "mllm" and query_text:
        from .rerank.mllm import rerank_mllm
        results = await rerank_mllm(
            results,
            query=query_text,
            top=rerank_flags.get("mllm_top", 10),
            timeout_ms=rerank_flags.get("timeout_ms", 5000),
            fallback_to_previous=rerank_flags.get("fallback_to_previous", True),
            trace_id=trace_id,
        )

    return results


async def run_search_multi_event(
    milvus,
    siglip2_encoder,
    bm25_index,
    events: list[dict],
    query_vectors: list[list[float]],
    answer_event: int = 0,
    upvote_ids: list[str] | None = None,
    downvote_ids: list[str] | None = None,
    flags: dict | None = None,
    filter_expr: str | None = None,
    trace_id: str = "",
) -> list[dict]:
    """Multi-event pipeline: per-event search → temporal → rerank.

    Args:
        events: List of event dicts with text_en, ocr, objects.
        query_vectors: One SigLIP2 text vector per event.
        answer_event: Index of the event containing the answer.
        Other args same as run_search.

    Returns:
        Ranked result list from the answer event.
    """
    from .retrieve.dense import search_dense
    from .retrieve.bm25 import search_bm25
    from .retrieve.fusion import rrf_fuse, apply_downvote_penalty

    if flags is None:
        from .config_flags import get_defaults
        flags = get_defaults()

    search_flags = flags.get("search", {})
    manual_flags = flags.get("manual", {})
    rerank_flags = flags.get("rerank", {})
    temporal_flags = flags.get("temporal", {})
    top_n = search_flags.get("top_n", 200)
    rrf_k = search_flags.get("rrf_k", 60)

    # Run per-event search — each event gets its own RRF
    variant_search = search_flags.get("variant_search", True)
    per_event_results: list[list[dict]] = []

    for eidx, (event, qvec) in enumerate(zip(events, query_vectors)):
        rrf_lists: list[tuple[str, list]] = []

        # Dense: primary vector + variant vectors
        if search_flags.get("dense", True):
            variants = event.get("variants", []) if variant_search else []
            all_vecs = [qvec]
            if variants:
                text_en = event.get("text_en", "")
                all_vecs = _encode_variants(siglip2_encoder, text_en, variants)

            with span(trace_id, f"retrieve.dense.e{eidx}",
                       n_vectors=len(all_vecs)) as attrs:
                dense_hits = search_dense(
                    milvus, all_vecs, top_n=top_n,
                    filter_expr=filter_expr, trace_id=trace_id,
                )
                for vi, hits in enumerate(dense_hits):
                    if hits:
                        label = f"dense_e{eidx}" if vi == 0 else f"variant_e{eidx}_v{vi}"
                        rrf_lists.append((label, hits))
                attrs["n_rrf_lists"] = len(rrf_lists)

        # BM25 on OCR terms from this event
        ocr_terms = event.get("ocr", [])
        if (search_flags.get("bm25", False)
                and bm25_index is not None
                and bm25_index.ready
                and ocr_terms):
            with span(trace_id, f"retrieve.bm25.e{eidx}"):
                bm25_hits = search_bm25(
                    bm25_index, ocr_terms, top_n=top_n, trace_id=trace_id,
                )
                for bi, hits in enumerate(bm25_hits):
                    if hits:
                        rrf_lists.append((f"bm25_e{eidx}_{bi}", hits))

        # Upvote (only for answer_event)
        if upvote_ids and eidx == answer_event:
            upvote_vecs = milvus.get_vectors(upvote_ids, "text_vector")
            for uid, vec in upvote_vecs.items():
                up_hits = milvus.search_batch(
                    vectors=[vec], anns_field="text_vector",
                    top_k=top_n, filter_expr=filter_expr,
                )
                if up_hits and up_hits[0]:
                    rrf_lists.append((f"upvote_{uid}", up_hits[0]))

        if rrf_lists:
            with span(trace_id, f"fusion.rrf.e{eidx}",
                       n_lists=len(rrf_lists), event_id=event.get("id", f"E{eidx+1}")):
                event_results = rrf_fuse(rrf_lists, k=rrf_k, top_n=top_n, trace_id=trace_id)
            _backfill_metadata(event_results, milvus)
        else:
            event_results = []

        per_event_results.append(event_results)

    # S4: Temporal search (before rerank) — gate on requires_temporal, not event count
    requires_temporal = temporal_flags.get("requires_temporal", False)
    if (temporal_flags.get("enabled", True)
            and requires_temporal
            and len(per_event_results) >= 2
            and len(query_vectors) >= 2):
        from .retrieve.temporal import temporal_search
        results = temporal_search(
            per_event_results=per_event_results,
            query_vectors=query_vectors,
            milvus=milvus,
            answer_event=answer_event,
            flags=temporal_flags,
            trace_id=trace_id,
        )
        if results:
            _backfill_metadata(results, milvus)
        else:
            results = per_event_results[answer_event] if per_event_results else []
    else:
        results = per_event_results[answer_event] if per_event_results else []

    # S3: Metadata soft boost (after temporal/RRF, before rerank)
    boost_flags = flags.get("metadata_boost", {})
    if boost_flags.get("enabled", False) and events:
        from .retrieve.metadata_boost import apply_metadata_boost
        answer_text = events[answer_event].get("text_en", "")
        results = apply_metadata_boost(
            results,
            query_terms=answer_text.lower().split() if answer_text else [],
            alpha=boost_flags.get("alpha", 0.05),
            max_boost=boost_flags.get("max_boost", 0.1),
            trace_id=trace_id,
        )

    # Downvote penalty
    if downvote_ids:
        dv_vecs_map = milvus.get_vectors(downvote_ids, "text_vector")
        dv_vecs = list(dv_vecs_map.values())
        if dv_vecs:
            penalty = manual_flags.get("downvote_penalty", 0.5)
            results = apply_downvote_penalty(results, dv_vecs, milvus, penalty=penalty)

    # S5: Rerank
    rerank_method = rerank_flags.get("method", "mllm")
    if rerank_method == "sg+mllm" and query_vectors:
        from .rerank.superglobal import rerank_superglobal
        from .rerank.mllm import rerank_mllm

        with span(trace_id, "rerank.sg_mllm") as attrs:
            sg_task = asyncio.to_thread(
                rerank_superglobal,
                list(results),
                query_vec=query_vectors[answer_event],
                milvus=milvus,
                top=rerank_flags.get("top", 50),
                m=rerank_flags.get("sg_m", 3),
                trace_id=trace_id,
            )
            scene = events[answer_event].get("text_en", "") if events else ""
            if scene:
                mllm_task = rerank_mllm(
                    list(results),
                    query=scene,
                    top=rerank_flags.get("mllm_top", 10),
                    timeout_ms=rerank_flags.get("timeout_ms", 5000),
                    fallback_to_previous=True,
                    trace_id=trace_id,
                )
                sg_out, mllm_out = await asyncio.gather(sg_task, mllm_task)
                results = _fuse_rerank_rrf(sg_out, mllm_out)
                attrs["fusion"] = "sg+mllm"
            else:
                results = await sg_task
                attrs["fusion"] = "sg_only"

    elif rerank_method == "mllm" and events:
        from .rerank.mllm import rerank_mllm
        scene = events[answer_event].get("text_en", "")
        if scene:
            results = await rerank_mllm(
                results,
                query=scene,
                top=rerank_flags.get("mllm_top", 10),
                timeout_ms=rerank_flags.get("timeout_ms", 5000),
                fallback_to_previous=rerank_flags.get("fallback_to_previous", True),
                trace_id=trace_id,
            )
    elif rerank_method == "superglobal" and query_vectors:
        from .rerank.superglobal import rerank_superglobal
        results = rerank_superglobal(
            results,
            query_vec=query_vectors[answer_event],
            milvus=milvus,
            top=rerank_flags.get("top", 50),
            m=rerank_flags.get("sg_m", 3),
            trace_id=trace_id,
        )
    elif rerank_method == "vlm" and events:
        from .rerank.vlm import rerank_vlm
        scene = events[answer_event].get("text_en", "")
        if scene:
            results = await rerank_vlm(
                results,
                scene_description=scene,
                top=rerank_flags.get("vlm_top", 20),
                timeout_ms=rerank_flags.get("timeout_ms", 2000),
                trace_id=trace_id,
            )

    return results


def _backfill_metadata(results: list[dict], milvus) -> None:
    """Fill in Milvus metadata for results that lack it (BM25/upvote-only)."""
    ids_needing_meta = [r["id"] for r in results if "video_name" not in r]
    if not ids_needing_meta:
        return
    meta_rows = milvus.get_by_ids(ids_needing_meta)
    meta_by_id = {m["id"]: m for m in meta_rows}
    for r in results:
        if "video_name" not in r and r["id"] in meta_by_id:
            m = meta_by_id[r["id"]]
            for field in ("video_name", "keyframe_idx", "frame_idx",
                          "pts_time", "fps", "objects", "ocr"):
                if field in m:
                    r[field] = m[field]

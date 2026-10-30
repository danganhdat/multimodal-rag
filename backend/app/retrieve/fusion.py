"""Client-side weighted RRF fusion.

Merges ranked lists from dense search, BM25, and upvote signals
into a single ranked list. Each source contributes 1/(k + rank + 1)
per item, optionally weighted.
"""

from collections import defaultdict

from ..observe import span


def rrf_fuse(
    lists: list[tuple[str, list]],
    k: int = 60,
    top_n: int = 200,
    trace_id: str = "",
) -> list[dict]:
    """Fuse multiple ranked lists using weighted RRF.

    Args:
        lists: [(source_name, hits)] where hits is a list of dicts with "id"
               and "score", or tuples (id, score). Ordered by rank.
        k: RRF constant (default 60).
        top_n: Max results to return.
        trace_id: For observability.

    Returns:
        List of dicts with "id", "rrf_score", and "sources" showing
        which source lists contributed and at what rank.
    """
    score_map: dict[str, float] = defaultdict(float)
    sources_map: dict[str, dict[str, int]] = defaultdict(dict)
    meta_map: dict[str, dict] = {}

    with span(trace_id, "fusion.rrf", n_lists=len(lists), k=k) as attrs:
        for source_name, hits in lists:
            for rank, hit in enumerate(hits):
                # Support both dict and tuple formats
                if isinstance(hit, dict):
                    hid = hit["id"]
                    if hid not in meta_map:
                        meta_map[hid] = hit
                elif isinstance(hit, (list, tuple)):
                    hid = hit[0]
                else:
                    continue

                score_map[hid] += 1.0 / (k + rank + 1)
                sources_map[hid][source_name] = rank

        # Sort by RRF score descending
        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_n]

        results = []
        for hid, rrf_score in ranked:
            entry = {
                "id": hid,
                "rrf_score": rrf_score,
                "sources": dict(sources_map[hid]),
            }
            # Copy metadata from the first source that had it
            if hid in meta_map:
                for field in ("video_name", "keyframe_idx", "frame_idx", "pts_time",
                              "fps", "objects", "ocr", "score"):
                    if field in meta_map[hid]:
                        entry[field] = meta_map[hid][field]
            results.append(entry)

        attrs["n_results"] = len(results)

    return results


def apply_downvote_penalty(
    results: list[dict],
    downvote_vectors: list[list[float]],
    milvus,
    penalty: float = 0.5,
    cos_threshold: float = 0.92,
) -> list[dict]:
    """Penalize results similar to downvoted frames.

    For each downvoted frame's vector, if a result has cos > threshold
    with that vector, multiply its score by penalty.
    """
    if not downvote_vectors or not results:
        return results

    import numpy as np

    result_ids = [r["id"] for r in results]
    result_vecs = milvus.get_vectors(result_ids, "text_vector")

    for r in results:
        vec = result_vecs.get(r["id"])
        if vec is None:
            continue
        v = np.array(vec)
        v_norm = v / (np.linalg.norm(v) + 1e-8)

        for dv in downvote_vectors:
            d = np.array(dv)
            d_norm = d / (np.linalg.norm(d) + 1e-8)
            cos_sim = float(np.dot(v_norm, d_norm))
            if cos_sim > cos_threshold:
                r["rrf_score"] *= penalty
                r.setdefault("sources", {})["downvote_penalty"] = round(cos_sim, 3)
                break

    # Re-sort after penalty
    results.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
    return results

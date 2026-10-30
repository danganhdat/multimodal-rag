"""S3: Object / metadata / sketch filter — default OFF.

When enabled, post-filters RRF results to keep only frames matching criteria.
"""

from ..observe import span


def apply_object_filter(
    results: list[dict],
    required_objects: list[str],
    trace_id: str = "",
) -> list[dict]:
    """Keep only results that contain at least one required COCO object."""
    if not required_objects:
        return results

    required = set(o.lower() for o in required_objects)
    with span(trace_id, "filter.objects", required=list(required)) as attrs:
        filtered = [
            r for r in results
            if required.intersection(o.lower() for o in r.get("objects", []))
        ]
        attrs["before"] = len(results)
        attrs["after"] = len(filtered)
    return filtered


def apply_metadata_filter(
    results: list[dict],
    video_names: list[str] | None = None,
    trace_id: str = "",
) -> list[dict]:
    """Keep only results from specified videos."""
    if not video_names:
        return results

    allowed = set(video_names)
    with span(trace_id, "filter.metadata", videos=video_names) as attrs:
        filtered = [r for r in results if r.get("video_name") in allowed]
        attrs["before"] = len(results)
        attrs["after"] = len(filtered)
    return filtered


def apply_sketch_reorder(
    results: list[dict],
    sketch_vector: list[float],
    milvus,
    trace_id: str = "",
) -> list[dict]:
    """Reorder results by cosine similarity to a sketch embedding on clip_vector."""
    import numpy as np

    if not results or not sketch_vector:
        return results

    ids = [r["id"] for r in results[:50]]
    with span(trace_id, "filter.sketch", n_items=len(ids)) as attrs:
        vecs_map = milvus.get_vectors(ids, "clip_vector")
        if not vecs_map:
            attrs["fallback"] = "no_clip_vectors"
            return results

        q = np.array(sketch_vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)

        scores = {}
        for fid in ids:
            if fid in vecs_map:
                v = np.array(vecs_map[fid], dtype=np.float32)
                v = v / (np.linalg.norm(v) + 1e-8)
                scores[fid] = float(q @ v)

        reranked = sorted(results[:50], key=lambda r: scores.get(r["id"], 0), reverse=True)
        return reranked + results[50:]

"""Temporal chain matching — greedy (mandatory) + adaptive tolerance (ablation).

Both strategies:
  - Same-video constraint: all events in a chain must be from the same video
  - Correct temporal order: timestamps must respect temporal_relations
  - Identical output schema for ablation comparison

Input:  per-event candidates C_1..C_N (with video_id, kf_id, timestamp_s, score)
Output: ranked chains [{video_id, frames, chain_score, coverage, partial}]
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..observe import span


def temporal_search(
    per_event_results: list[list[dict]],
    query_vectors: list[list[float]],
    milvus,
    answer_event: int = 0,
    flags: dict | None = None,
    trace_id: str = "",
) -> list[dict]:
    """Dispatch to greedy or adaptive_tolerance strategy.

    Returns answer-event results scored by chain quality, or empty list
    if no valid chains found.
    """
    tf = flags or {}
    strategy = tf.get("strategy", "greedy")
    n_events = len(per_event_results)

    if n_events < 2:
        return per_event_results[0] if per_event_results else []

    max_candidates = tf.get("max_candidates_per_event", 200)
    top_chains = tf.get("top_chains", 20)

    with span(trace_id, "temporal.search", strategy=strategy,
              n_events=n_events, top_chains=top_chains) as attrs:

        # Build per-event candidate lists keyed by video
        video_candidates = _group_by_video(per_event_results, max_candidates)

        if not video_candidates:
            attrs["n_chains"] = 0
            return []

        # Precompute similarity scores for adaptive tolerance
        q_vecs = None
        if strategy == "adaptive_tolerance":
            q_vecs = _normalize_vectors(query_vectors)

        # Find chains per video
        all_chains = []
        for video_id, event_cands in video_candidates.items():
            if strategy == "adaptive_tolerance":
                chains = _adaptive_tolerance_chains(
                    video_id, event_cands, n_events, milvus, q_vecs,
                    tolerance_threshold=tf.get("tolerance_threshold", 3),
                    max_gap_s=tf.get("max_gap_s"),
                )
            else:
                chains = _greedy_chains(
                    video_id, event_cands, n_events,
                    max_gap_s=tf.get("max_gap_s"),
                )
            all_chains.extend(chains)

        # Score + rank + deduplicate
        all_chains.sort(key=lambda c: c["chain_score"], reverse=True)
        deduped = _deduplicate_chains(all_chains, answer_event)
        top = deduped[:top_chains]

        attrs["n_chains"] = len(top)
        attrs["n_videos_searched"] = len(video_candidates)

    return _chains_to_results(top, answer_event)


# ---------------------------------------------------------------------------
# Strategy A — Greedy (mandatory V1)
# ---------------------------------------------------------------------------

def _greedy_chains(
    video_id: str,
    event_cands: dict[int, list[dict]],
    n_events: int,
    max_gap_s: float | None = None,
) -> list[dict]:
    """Greedy: anchor on first event, pick best-scoring candidate per subsequent event."""
    chains = []

    anchors = event_cands.get(0, [])
    for anchor in anchors[:50]:
        frames = [None] * n_events
        frames[0] = anchor
        prev_ts = anchor["pts_time"]

        for eidx in range(1, n_events):
            cands = event_cands.get(eidx, [])
            best = None
            best_score = -1.0

            for c in cands:
                # Temporal order: must come after previous event
                if c["pts_time"] <= prev_ts:
                    continue
                # Optional gap constraint
                if max_gap_s is not None and (c["pts_time"] - prev_ts) > max_gap_s:
                    continue
                if c["score"] > best_score:
                    best = c
                    best_score = c["score"]

            if best is not None:
                frames[eidx] = best
                prev_ts = best["pts_time"]

        chain = _build_chain(video_id, frames, n_events)
        if chain is not None:
            chains.append(chain)

    return chains


# ---------------------------------------------------------------------------
# Strategy B — Adaptive Tolerance (ablation)
# ---------------------------------------------------------------------------

def _adaptive_tolerance_chains(
    video_id: str,
    event_cands: dict[int, list[dict]],
    n_events: int,
    milvus,
    q_vecs: list[np.ndarray] | None,
    tolerance_threshold: int = 3,
    max_gap_s: float | None = None,
) -> list[dict]:
    """Adaptive tolerance: early-stopping per event based on similarity plateau."""
    chains = []

    anchors = event_cands.get(0, [])
    for anchor in anchors[:50]:
        frames = [None] * n_events
        frames[0] = anchor
        prev_ts = anchor["pts_time"]

        for eidx in range(1, n_events):
            cands = event_cands.get(eidx, [])
            # Filter to valid temporal range
            valid = [
                c for c in cands
                if c["pts_time"] > prev_ts
                and (max_gap_s is None or (c["pts_time"] - prev_ts) <= max_gap_s)
            ]

            if not valid:
                continue

            # Adaptive tolerance selection
            best_score = -1.0
            best_cand = None
            tolerance = 0

            for c in valid:
                score = c["score"]
                # If we have query vectors, use cosine similarity
                if q_vecs is not None and eidx < len(q_vecs):
                    vec = c.get("_vector")
                    if vec is not None:
                        v = np.array(vec, dtype=np.float32)
                        norm = np.linalg.norm(v)
                        if norm > 1e-8:
                            score = float(np.dot(q_vecs[eidx], v / norm))

                if score > best_score:
                    best_score = score
                    best_cand = c
                    tolerance = 0
                else:
                    tolerance += 1

                if tolerance >= tolerance_threshold:
                    break

            if best_cand is not None:
                frames[eidx] = best_cand
                prev_ts = best_cand["pts_time"]

        chain = _build_chain(video_id, frames, n_events)
        if chain is not None:
            chains.append(chain)

    return chains


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _group_by_video(
    per_event_results: list[list[dict]],
    max_per_event: int = 200,
) -> dict[str, dict[int, list[dict]]]:
    """Group candidates by video_id, keeping per-event lists sorted by score."""
    video_cands: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for eidx, results in enumerate(per_event_results):
        for r in results[:max_per_event]:
            vname = r.get("video_name", "")
            if not vname:
                continue
            video_cands[vname][eidx].append({
                "kf_id": r.get("id", ""),
                "video_id": vname,
                "keyframe_idx": r.get("keyframe_idx", 0),
                "pts_time": r.get("pts_time", 0.0),
                "frame_idx": r.get("frame_idx", 0),
                "score": r.get("rrf_score", r.get("score", 0.0)),
                "fps": r.get("fps", 25.0),
            })

    # Only keep videos that have candidates for at least 2 events
    return {
        vid: dict(events)
        for vid, events in video_cands.items()
        if len(events) >= 2
    }


def _build_chain(
    video_id: str,
    frames: list[dict | None],
    n_events: int,
) -> dict | None:
    """Build a chain dict from frame selections. Returns None if too few events covered."""
    found = [f for f in frames if f is not None]
    coverage = [f"E{i+1}" for i, f in enumerate(frames) if f is not None]
    partial = len(found) < n_events

    # Require at least n_events - 1 covered
    if len(found) < max(2, n_events - 1):
        return None

    # Check temporal order
    timestamps = [f["pts_time"] for f in found]
    if timestamps != sorted(timestamps):
        return None

    # Compute chain score: mean relevance * coverage ratio
    mean_score = sum(f["score"] for f in found) / len(found)
    coverage_ratio = len(found) / n_events
    chain_score = mean_score * coverage_ratio

    chain_frames = []
    for i, f in enumerate(frames):
        if f is not None:
            chain_frames.append({
                "event_id": f"E{i+1}",
                "kf_id": f["kf_id"],
                "timestamp_s": f["pts_time"],
                "score": f["score"],
            })

    return {
        "video_id": video_id,
        "frames": chain_frames,
        "chain_score": chain_score,
        "coverage": coverage,
        "partial": partial,
    }


def _deduplicate_chains(
    chains: list[dict],
    answer_event: int,
) -> list[dict]:
    """Keep best chain per answer-event kf_id."""
    best: dict[str, dict] = {}
    for c in chains:
        answer_frame = None
        for f in c["frames"]:
            if f["event_id"] == f"E{answer_event + 1}":
                answer_frame = f["kf_id"]
                break
        if answer_frame is None:
            continue
        key = answer_frame
        if key not in best or c["chain_score"] > best[key]["chain_score"]:
            best[key] = c
    return sorted(best.values(), key=lambda c: c["chain_score"], reverse=True)


def _chains_to_results(chains: list[dict], answer_event: int) -> list[dict]:
    """Convert chains to the result dict format expected by the pipeline."""
    results = []
    for c in chains:
        answer_frame = None
        for f in c["frames"]:
            if f["event_id"] == f"E{answer_event + 1}":
                answer_frame = f
                break
        if answer_frame is None:
            continue

        results.append({
            "id": answer_frame["kf_id"],
            "video_name": c["video_id"],
            "keyframe_idx": int(answer_frame["kf_id"].split("/")[-1]) if "/" in answer_frame["kf_id"] else 0,
            "pts_time": answer_frame["timestamp_s"],
            "rrf_score": c["chain_score"],
            "chain_score": c["chain_score"],
            "chain": {f["event_id"]: f["kf_id"] for f in c["frames"]},
            "coverage": c["coverage"],
            "partial": c["partial"],
            "sources": {"temporal": c["chain_score"]},
        })
    return results


def _normalize_vectors(vectors: list[list[float]]) -> list[np.ndarray]:
    """Normalize query vectors for cosine similarity."""
    result = []
    for v in vectors:
        a = np.array(v, dtype=np.float32)
        norm = np.linalg.norm(a)
        if norm > 1e-8:
            a = a / norm
        result.append(a)
    return result

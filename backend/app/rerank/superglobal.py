"""SuperGlobal reranker: mean of frame + m nearest neighbors within top-N.

No model, no API call — just numpy on vectors already in Milvus.
For each frame, compute v_hat = normalize(mean(v_i, m closest neighbors
in the top-N set)), then rescore as cos(query, v_hat).
"""

import numpy as np

from ..observe import span


def rerank_superglobal(
    results: list[dict],
    query_vec: list[float],
    milvus,
    top: int = 50,
    m: int = 3,
    trace_id: str = "",
) -> list[dict]:
    """Rerank top results using SuperGlobal aggregation.

    Args:
        results: RRF-fused result dicts (must have 'id' and 'rrf_score').
        query_vec: The query text vector (SigLIP2).
        milvus: MilvusSearchService for fetching vectors.
        top: How many results to rerank (rest keep their order after).
        m: Number of nearest neighbors to aggregate with.
        trace_id: For observability.

    Returns:
        Reranked list of result dicts with updated scores.
    """
    if len(results) <= 1:
        return results

    to_rerank = results[:top]
    rest = results[top:]
    ids = [r["id"] for r in to_rerank]

    with span(trace_id, "rerank.superglobal", top=top, m=m, n_items=len(ids)) as attrs:
        vecs_map = milvus.get_vectors(ids, "text_vector")
        if not vecs_map:
            attrs["fallback"] = "no_vectors"
            return results

        q = np.array(query_vec, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-8)

        # Build matrix of vectors for the top-N set
        ordered_ids = [i for i in ids if i in vecs_map]
        if len(ordered_ids) < 2:
            return results

        V = np.array([vecs_map[i] for i in ordered_ids], dtype=np.float32)

        # Normalize rows
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        V_norm = V / norms

        # Pairwise cosine similarity within the set
        sim_matrix = V_norm @ V_norm.T

        # For each frame, find m nearest neighbors (excluding self)
        new_scores = {}
        for idx, fid in enumerate(ordered_ids):
            sims = sim_matrix[idx].copy()
            sims[idx] = -1.0  # exclude self
            if m >= len(ordered_ids) - 1:
                neighbor_idxs = [j for j in range(len(ordered_ids)) if j != idx]
            else:
                neighbor_idxs = np.argpartition(sims, -m)[-m:]

            # Aggregate: mean of self + neighbors
            vecs_to_avg = [V_norm[idx]]
            for ni in neighbor_idxs:
                vecs_to_avg.append(V_norm[ni])
            v_hat = np.mean(vecs_to_avg, axis=0)
            v_hat = v_hat / (np.linalg.norm(v_hat) + 1e-8)

            new_scores[fid] = float(q_norm @ v_hat)

        # Sort reranked portion by new score
        for r in to_rerank:
            if r["id"] in new_scores:
                r["sources"]["rerank"] = new_scores[r["id"]]
        to_rerank.sort(key=lambda r: new_scores.get(r["id"], 0), reverse=True)

        attrs["score_range"] = f"{min(new_scores.values()):.4f}-{max(new_scores.values()):.4f}"

    return to_rerank + rest

"""Metadata soft boost: multiplicative score adjustment based on metadata signals.

boost = rrf_score * (1 + alpha * relevance), capped at max_boost.
Signals: title/description Jaccard overlap with query terms.
"""

from __future__ import annotations

from ..observe import span


def apply_metadata_boost(
    results: list[dict],
    query_terms: list[str],
    alpha: float = 0.05,
    max_boost: float = 0.1,
    trace_id: str = "",
) -> list[dict]:
    """Apply multiplicative metadata boost to search results.

    Args:
        results: Ranked results with rrf_score.
        query_terms: Lowercased query words for Jaccard overlap.
        alpha: Boost strength multiplier.
        max_boost: Maximum multiplicative boost (caps alpha * relevance).
        trace_id: For observability.

    Returns:
        Results with adjusted rrf_score, re-sorted.
    """
    if not results or not query_terms or alpha <= 0:
        return results

    query_set = set(query_terms)

    with span(trace_id, "metadata.boost", n_candidates=len(results),
              alpha=alpha, max_boost=max_boost) as attrs:
        n_boosted = 0
        max_delta = 0.0

        for r in results:
            relevance = _compute_relevance(r, query_set)
            if relevance > 0:
                boost_factor = min(alpha * relevance, max_boost)
                old_score = r.get("rrf_score", 0.0)
                r["rrf_score"] = old_score * (1 + boost_factor)
                r.setdefault("sources", {})["metadata_boost"] = boost_factor
                delta = r["rrf_score"] - old_score
                max_delta = max(max_delta, delta)
                n_boosted += 1

        results.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)

        attrs["n_boosted"] = n_boosted
        attrs["max_delta"] = round(max_delta, 6)

    return results


def _compute_relevance(result: dict, query_set: set[str]) -> float:
    """Jaccard overlap between query terms and result metadata tokens."""
    meta_tokens: set[str] = set()

    for obj in result.get("objects", []):
        meta_tokens.update(obj.lower().split())

    ocr = result.get("ocr", "")
    if ocr:
        meta_tokens.update(ocr.lower().split())

    if not meta_tokens:
        return 0.0

    intersection = query_set & meta_tokens
    union = query_set | meta_tokens
    return len(intersection) / len(union) if union else 0.0

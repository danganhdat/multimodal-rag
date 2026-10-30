"""MLLM listwise reranking — shared capability for Manual and Agentic modes.

Shortlists top-N candidates, sends frame images + query to VLM for listwise
ranking, validates returned IDs, falls back to previous stage on error/timeout.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from ..config import settings
from ..observe import span


async def rerank_mllm(
    results: list[dict],
    query: str,
    top: int = 10,
    timeout_ms: int = 5000,
    fallback_to_previous: bool = True,
    trace_id: str = "",
) -> list[dict]:
    """MLLM listwise reranking of top candidates.

    Sends shortlisted frame images to VLM with the query. VLM returns
    a ranked list of kf_ids from most to least relevant.

    Args:
        results: Ranked results from prior stage (RRF/temporal).
        query: Scene description for relevance scoring.
        top: Number of candidates to shortlist for MLLM.
        timeout_ms: Timeout for MLLM call in milliseconds.
        fallback_to_previous: Return original results on error.
        trace_id: For observability.

    Returns:
        Re-ranked results list.
    """
    if not results or top <= 0:
        return results

    with span(trace_id, "rerank.mllm", n_candidates=len(results),
              shortlist=top, query=query[:200]) as attrs:

        shortlist = results[:top]
        remainder = results[top:]

        # Collect image paths for shortlisted candidates
        keyframes_dir = Path(settings.KEYFRAMES_DIR)
        candidates = []
        for r in shortlist:
            kf_id = r.get("id", "")
            parts = kf_id.split("/")
            if len(parts) == 2:
                img = keyframes_dir / parts[0] / f"{parts[1]}.jpg"
                if img.exists():
                    candidates.append({"id": kf_id, "path": str(img), "result": r})

        if len(candidates) < 2:
            attrs["status"] = "too_few_candidates"
            return results

        try:
            from ..llm import get_chat, image_message, LLMUnavailable

            chat = get_chat("vlm", temperature=0, max_tokens=1000, timeout=timeout_ms / 1000)

            prompt = _build_rerank_prompt(query, [c["id"] for c in candidates])
            image_paths = [c["path"] for c in candidates]
            msg = image_message(prompt, image_paths, max_side=384)

            response = await asyncio.wait_for(
                chat.ainvoke([msg]),
                timeout=timeout_ms / 1000,
            )

            ranked_ids = _parse_ranking(response.content, {c["id"] for c in candidates})
            attrs["n_ranked"] = len(ranked_ids)

            if not ranked_ids:
                attrs["status"] = "parse_failed"
                if fallback_to_previous:
                    return results
                return results

            # Build reranked list
            result_by_id = {c["id"]: c["result"] for c in candidates}
            reranked = []
            seen = set()

            # First: MLLM-ranked items in order
            for i, kf_id in enumerate(ranked_ids):
                if kf_id in result_by_id and kf_id not in seen:
                    r = dict(result_by_id[kf_id])
                    r["mllm_rank"] = i + 1
                    r.setdefault("sources", {})["mllm_rerank"] = len(ranked_ids) - i
                    reranked.append(r)
                    seen.add(kf_id)

            # Then: shortlisted items MLLM didn't rank (preserve original order)
            for c in candidates:
                if c["id"] not in seen:
                    reranked.append(c["result"])
                    seen.add(c["id"])

            # Finally: remainder
            reranked.extend(remainder)

            attrs["status"] = "success"
            return reranked

        except LLMUnavailable:
            attrs["status"] = "vlm_unavailable"
            return results
        except asyncio.TimeoutError:
            attrs["status"] = "timeout"
            if fallback_to_previous:
                return results
            return results
        except Exception as e:
            attrs["status"] = f"error:{type(e).__name__}"
            if fallback_to_previous:
                return results
            return results


def _build_rerank_prompt(query: str, candidate_ids: list[str]) -> str:
    """Build the listwise reranking prompt."""
    id_list = "\n".join(f"  {i+1}. {cid}" for i, cid in enumerate(candidate_ids))
    return (
        f"You are a visual search reranker. Given a query and candidate keyframe images, "
        f"rank them from MOST to LEAST relevant to the query.\n\n"
        f"Query: {query}\n\n"
        f"Candidate IDs (in the same order as the images):\n{id_list}\n\n"
        f"Return ONLY a JSON array of the candidate IDs ranked from most to least relevant. "
        f"Include ALL candidate IDs. Example: [\"L01_V001/042\", \"L01_V001/015\", ...]\n"
        f"Do not include any other text."
    )


def _parse_ranking(text: str, valid_ids: set[str]) -> list[str]:
    """Parse MLLM response into a list of valid kf_ids."""
    text = text.strip()

    # Try to extract JSON array
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(x) for x in parsed if str(x) in valid_ids]
        except json.JSONDecodeError:
            pass

    # Fallback: look for kf_id patterns in order
    pattern = re.compile(r'L\d+_V\d+/\d{3,}')
    found = pattern.findall(text)
    return [x for x in found if x in valid_ids]

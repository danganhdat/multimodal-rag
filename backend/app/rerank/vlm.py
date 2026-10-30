"""VLM reranker: use vision-language model to score frame relevance.

Sends top-K frames to VLM in batches, asks for 0-10 relevance score
against the scene description. Falls back to tier-1 order on timeout/error.
"""

import asyncio
from pathlib import Path

from ..config import settings
from ..observe import span


async def rerank_vlm(
    results: list[dict],
    scene_description: str,
    top: int = 20,
    batch_size: int = 4,
    timeout_ms: int = 2000,
    trace_id: str = "",
) -> list[dict]:
    """Rerank top results using VLM relevance scoring.

    Args:
        results: Result dicts from tier-1 (must have 'id', 'video_name', 'keyframe_idx').
        scene_description: Visual scene description (not the question).
        top: How many results to rerank via VLM.
        batch_size: Frames per VLM request.
        timeout_ms: Total timeout; on expiry, return tier-1 order.
        trace_id: For observability.

    Returns:
        Reranked list with VLM scores in sources['rerank'].
    """
    from ..llm import get_chat, image_message, LLMUnavailable

    to_rerank = results[:top]
    rest = results[top:]

    with span(trace_id, "rerank.vlm", top=top, timeout_ms=timeout_ms) as attrs:
        try:
            chat = get_chat("vlm", temperature=0)
        except LLMUnavailable:
            attrs["fallback"] = "no_vlm"
            return results

        keyframes_dir = Path(settings.KEYFRAMES_DIR)

        async def score_batch(batch: list[dict]) -> list[tuple[str, float]]:
            image_paths = []
            for r in batch:
                vname = r.get("video_name", "")
                kidx = r.get("keyframe_idx", 0)
                img_path = keyframes_dir / vname / f"{kidx:03d}.jpg"
                if img_path.exists():
                    image_paths.append(str(img_path))
                else:
                    image_paths.append(None)

            valid_paths = [p for p in image_paths if p is not None]
            if not valid_paths:
                return [(r["id"], 5.0) for r in batch]

            prompt = (
                f"Rate how well each image matches this scene description "
                f"on a scale of 0-10 (0=completely irrelevant, 10=perfect match).\n"
                f"Scene: {scene_description}\n\n"
                f"Reply with ONLY a comma-separated list of scores, "
                f"one per image, in order. Example: 8,3,7,5"
            )
            msg = image_message(prompt, valid_paths, max_side=512)
            response = await chat.ainvoke([msg])

            raw = response.content.strip()
            try:
                scores = [float(s.strip()) for s in raw.split(",")]
            except ValueError:
                scores = [5.0] * len(batch)

            result_scores = []
            score_idx = 0
            for i, r in enumerate(batch):
                if image_paths[i] is not None and score_idx < len(scores):
                    result_scores.append((r["id"], max(0, min(10, scores[score_idx]))))
                    score_idx += 1
                else:
                    result_scores.append((r["id"], 5.0))
            return result_scores

        try:
            # Split into batches and run
            batches = []
            for i in range(0, len(to_rerank), batch_size):
                batches.append(to_rerank[i : i + batch_size])

            all_scores: dict[str, float] = {}
            timeout_s = timeout_ms / 1000.0

            async def run_all():
                for batch in batches:
                    batch_scores = await score_batch(batch)
                    for fid, score in batch_scores:
                        all_scores[fid] = score

            await asyncio.wait_for(run_all(), timeout=timeout_s)

            for r in to_rerank:
                if r["id"] in all_scores:
                    r["sources"]["rerank"] = all_scores[r["id"]]

            # Sort by VLM score, then by original order for ties
            to_rerank.sort(
                key=lambda r: all_scores.get(r["id"], 5.0), reverse=True
            )
            attrs["scored"] = len(all_scores)
            attrs["fallback"] = False

        except (asyncio.TimeoutError, Exception) as e:
            attrs["fallback"] = True
            attrs["error"] = str(e)[:200]

    return to_rerank + rest

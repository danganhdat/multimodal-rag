"""S6: Shot validation via VLM — default OFF.

Sends shot images (keyframe + ±1 neighbors) to VLM with the scene description.
VLM answers yes/no whether the shot matches. Removes non-matching shots.
"""

import asyncio
from pathlib import Path

from ..config import settings
from ..observe import span


async def validate_shots(
    results: list[dict],
    scene_description: str,
    milvus,
    top: int = 10,
    trace_id: str = "",
) -> list[dict]:
    """Validate top results by asking VLM if the shot matches."""
    from ..llm import get_chat, image_message, LLMUnavailable

    to_check = results[:top]
    rest = results[top:]

    with span(trace_id, "shot_validation", top=top) as attrs:
        try:
            chat = get_chat("vlm", temperature=0)
        except LLMUnavailable:
            attrs["fallback"] = "no_vlm"
            return results

        keyframes_dir = Path(settings.KEYFRAMES_DIR)
        validated = []

        for r in to_check:
            vname = r.get("video_name", "")
            kidx = r.get("keyframe_idx", 0)
            img_path = keyframes_dir / vname / f"{kidx:03d}.jpg"
            if not img_path.exists():
                validated.append(r)
                continue

            prompt = (
                f"Does this keyframe match the following scene description? "
                f"Answer YES or NO only.\n\n"
                f"Scene: {scene_description}"
            )

            try:
                msg = image_message(prompt, [str(img_path)], max_side=512)
                response = await chat.ainvoke([msg])
                answer = response.content.strip().upper()
                if "YES" in answer:
                    validated.append(r)
            except Exception:
                validated.append(r)

        attrs["before"] = len(to_check)
        attrs["after"] = len(validated)

    return validated + rest

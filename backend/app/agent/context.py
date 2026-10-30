"""S7 (frame input) + S8 (shot summary) — default OFF.

S7: Load top-N frame images directly into the agent's initial prompt.
S8: VLM writes a short description per shot for the agent context.
"""

from pathlib import Path

from ..config import settings
from ..observe import span


def build_frame_context(
    kf_ids: list[str],
    max_frames: int = 3,
    trace_id: str = "",
) -> list[str]:
    """Return image paths for the top frames to embed in the agent prompt.

    Args:
        kf_ids: Keyframe IDs to include.
        max_frames: Maximum number of frames.

    Returns:
        List of file paths to existing images.
    """
    keyframes_dir = Path(settings.KEYFRAMES_DIR)
    paths = []
    for kf_id in kf_ids[:max_frames]:
        parts = kf_id.split("/")
        if len(parts) == 2:
            img = keyframes_dir / parts[0] / f"{parts[1]}.jpg"
            if img.exists():
                paths.append(str(img))
    return paths


async def summarize_shots(
    kf_ids: list[str],
    max_shots: int = 5,
    trace_id: str = "",
) -> dict[str, str]:
    """VLM writes a 1-2 sentence description of each keyframe.

    Returns {kf_id: summary_text}.
    """
    from ..llm import get_chat, image_message, LLMUnavailable

    with span(trace_id, "shot_summary", n_shots=len(kf_ids[:max_shots])) as attrs:
        try:
            chat = get_chat("vlm", temperature=0)
        except LLMUnavailable:
            attrs["fallback"] = "no_vlm"
            return {}

        keyframes_dir = Path(settings.KEYFRAMES_DIR)
        summaries = {}

        for kf_id in kf_ids[:max_shots]:
            parts = kf_id.split("/")
            if len(parts) != 2:
                continue
            img = keyframes_dir / parts[0] / f"{parts[1]}.jpg"
            if not img.exists():
                continue

            prompt = (
                "Describe this keyframe in 1-2 sentences. "
                "Focus on visible text, objects, people, and actions."
            )
            try:
                msg = image_message(prompt, [str(img)], max_side=512)
                response = await chat.ainvoke([msg])
                summaries[kf_id] = response.content.strip()[:300]
            except Exception:
                continue

        attrs["n_summarized"] = len(summaries)

    return summaries

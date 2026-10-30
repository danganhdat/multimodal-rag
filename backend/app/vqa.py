"""One-shot MLLM VQA answer generation.

Single MLLM call: evidence frames + question → answer.
Does NOT enter ReAct loop. Returns KHONG_TRA_LOI_DUOC when insufficient.
"""

from __future__ import annotations

from .evidence import EvidenceSet
from .observe import span

_INSUFFICIENT_ANSWER = "KHONG_TRA_LOI_DUOC"


async def generate_answer(
    question: str,
    evidence: EvidenceSet,
    trace_id: str = "",
    timeout_s: float = 60.0,
) -> dict:
    """Generate a VQA answer from evidence frames.

    Args:
        question: The user's question.
        evidence: EvidenceSet with frames and image paths.
        trace_id: For observability.
        timeout_s: Timeout for VLM call.

    Returns:
        {"status", "answer", "evidence", "usage"}
    """
    import asyncio

    with span(trace_id, "vqa.generate", question=question[:200],
              n_evidence=len(evidence.frames),
              sufficient=evidence.sufficient) as attrs:

        if not evidence.sufficient or not evidence.image_paths:
            attrs["status"] = "insufficient_evidence"
            return {
                "status": "insufficient_evidence",
                "answer": _INSUFFICIENT_ANSWER,
                "evidence": evidence.kf_ids,
                "usage": {},
            }

        try:
            from .llm import get_chat, image_message, LLMUnavailable

            chat = get_chat("vlm", temperature=0, max_tokens=500,
                            timeout=timeout_s)

            prompt = _build_prompt(question, evidence)
            msg = image_message(prompt, evidence.image_paths, max_side=768)

            response = await asyncio.wait_for(
                chat.ainvoke([msg]),
                timeout=timeout_s,
            )

            answer_text = response.content.strip()
            usage = {}
            if hasattr(response, "response_metadata"):
                meta = response.response_metadata
                usage = {
                    "tokens_in": meta.get("token_usage", {}).get("prompt_tokens", 0),
                    "tokens_out": meta.get("token_usage", {}).get("completion_tokens", 0),
                }

            if not answer_text or answer_text.upper() == _INSUFFICIENT_ANSWER:
                attrs["status"] = "insufficient_evidence"
                return {
                    "status": "insufficient_evidence",
                    "answer": _INSUFFICIENT_ANSWER,
                    "evidence": evidence.kf_ids,
                    "usage": usage,
                }

            attrs["status"] = "answered"
            attrs["answer_len"] = len(answer_text)
            return {
                "status": "answered",
                "answer": answer_text,
                "evidence": evidence.kf_ids,
                "usage": usage,
            }

        except LLMUnavailable as e:
            attrs["status"] = "error"
            return {
                "status": "error",
                "answer": f"VLM unavailable: {e}",
                "evidence": evidence.kf_ids,
                "usage": {},
            }
        except asyncio.TimeoutError:
            attrs["status"] = "timeout"
            return {
                "status": "error",
                "answer": "VQA generation timed out.",
                "evidence": evidence.kf_ids,
                "usage": {},
            }
        except Exception as e:
            attrs["status"] = f"error:{type(e).__name__}"
            return {
                "status": "error",
                "answer": f"VQA error: {e}",
                "evidence": evidence.kf_ids,
                "usage": {},
            }


def _build_prompt(question: str, evidence: EvidenceSet) -> str:
    """Build the VQA prompt for the VLM."""
    frame_descriptions = []
    for i, f in enumerate(evidence.frames):
        desc = f"Image {i+1}: keyframe {f.kf_id}"
        if f.event_id:
            desc += f" (event {f.event_id})"
        if f.timestamp_s > 0:
            desc += f" at {f.timestamp_s:.1f}s"
        if f.ocr:
            desc += f"\n  Text visible in frame: {f.ocr}"
        frame_descriptions.append(desc)

    frames_text = "\n".join(frame_descriptions)

    return (
        f"You are answering a visual question about video keyframes.\n\n"
        f"Evidence frames:\n{frames_text}\n\n"
        f"Question: {question}\n\n"
        f"Instructions:\n"
        f"- Answer based on what you can see in the frames AND the OCR text provided.\n"
        f"- Read and use any text, labels, signs, or captions visible in the frames.\n"
        f"- The 'Text visible in frame' lines contain OCR-extracted text — use this to answer text-reading questions.\n"
        f"- Be concise and specific.\n"
        f"- If the answer is a number, give just the number.\n"
        f"- If the answer is a name or label, give just that text.\n"
        f"- If you cannot determine the answer from the frames, respond with: {_INSUFFICIENT_ANSWER}\n"
        f"- Do NOT guess or hallucinate information not visible in the frames."
    )

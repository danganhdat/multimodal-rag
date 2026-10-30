"""Query Enhancement: LLM-based query parsing + regex pre-extraction.

One LLM call → structured ParsedQuery (events with text_en, ocr, objects).
Falls back to translation-only on LLM error.
"""

import json
from pathlib import Path

from pydantic import BaseModel

from ..config import settings
from ..observe import span

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "agent" / "prompts"


class EventQuery(BaseModel):
    id: str = ""
    text_vi: str = ""
    text_en: str = ""
    ocr: list[str] = []
    objects: list[str] = []
    variants: list[str] = []
    enabled: bool = True


class TemporalRelation(BaseModel):
    from_event: str = ""
    to_event: str = ""
    type: str = "before"


class ParsedQuery(BaseModel):
    classification: str = "single_scene"
    events: list[EventQuery]
    temporal_relations: list[TemporalRelation] = []
    requires_temporal: bool = False
    answer_event: int = 0
    question: str = ""
    answer_type: str = "text"
    variants: list[str] = []
    fallback: bool = False


def _assign_event_ids(parsed: ParsedQuery) -> None:
    """Assign event IDs if the LLM didn't provide them."""
    seen = set()
    for i, event in enumerate(parsed.events):
        if not event.id or event.id in seen:
            event.id = f"E{i + 1}"
        seen.add(event.id)


def _infer_requires_temporal(parsed: ParsedQuery) -> None:
    """Set requires_temporal based on temporal_relations, not event count."""
    if parsed.temporal_relations:
        event_ids = {e.id for e in parsed.events}
        valid = [
            r for r in parsed.temporal_relations
            if r.from_event in event_ids and r.to_event in event_ids
        ]
        parsed.temporal_relations = valid
        parsed.requires_temporal = len(valid) > 0
    else:
        parsed.requires_temporal = False


def _cap_variants(parsed: ParsedQuery, max_variants: int = 3) -> None:
    """Truncate per-event variants to max_variants."""
    for event in parsed.events:
        if len(event.variants) > max_variants:
            event.variants = event.variants[:max_variants]


def _load_prompt(version: str = "v1") -> str:
    """Load prompt template by version (v1 or v2)."""
    path = _PROMPTS_DIR / f"qe_{version}.txt"
    if not path.exists():
        path = _PROMPTS_DIR / "qe_v1.txt"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        obj_path = Path(__file__).resolve().parent.parent.parent.parent / "object_classes.json"
        if obj_path.exists():
            classes = json.loads(obj_path.read_text(encoding="utf-8"))
            text = text.replace("{COCO_CLASSES}", ", ".join(classes))
        else:
            text = text.replace("{COCO_CLASSES}", "(not available)")
        return text
    return "Parse the user question into search events. Return JSON."


async def parse_query(
    question: str,
    lang: str = "auto",
    flags: dict | None = None,
    trace_id: str = "",
) -> ParsedQuery:
    """Parse a user question into structured search events.

    Uses LLM if available, falls back to translation + regex.

    Args:
        question: The raw user question.
        lang: "vi", "en", or "auto" (detect).
        flags: Feature flags (qe section).
        trace_id: For observability.

    Returns:
        ParsedQuery with events, question, answer_type.
    """
    from .regex import regex_parse

    qe_flags = (flags or {}).get("qe", {})
    enabled = qe_flags.get("enabled", True)
    prompt_version = qe_flags.get("prompt_version", "v1")

    # Always run regex first (instant)
    regex_result = regex_parse(question)

    if not enabled:
        return _fallback_parse(question, lang, regex_result, trace_id)

    try:
        from ..llm import get_chat, LLMUnavailable

        with span(trace_id, "qe.parse", prompt_version=f"qe_{prompt_version}") as attrs:
            try:
                chat = get_chat("text", temperature=0)
            except LLMUnavailable:
                attrs["fallback"] = "no_api_key"
                return _fallback_parse(question, lang, regex_result, trace_id)

            prompt = _load_prompt(prompt_version)
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=question),
            ]

            response = await chat.ainvoke(messages)
            raw = response.content.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            data = json.loads(raw)

            # Normalize: strip unknown fields, provide defaults
            for ev in data.get("events", []):
                if "variants" not in ev:
                    ev["variants"] = []
                if "id" not in ev:
                    ev["id"] = ""
                if "enabled" not in ev:
                    ev["enabled"] = True
            if "classification" not in data:
                data["classification"] = "single_scene"
            if "temporal_relations" not in data:
                data["temporal_relations"] = []

            parsed = ParsedQuery(**data)

            _assign_event_ids(parsed)
            _infer_requires_temporal(parsed)
            _cap_variants(parsed, max_variants=qe_flags.get("max_variants", 3))

            for event in parsed.events:
                for ocr_item in regex_result.get("ocr", []):
                    if ocr_item not in event.ocr:
                        event.ocr.append(ocr_item)
                for obj_item in regex_result.get("objects", []):
                    if obj_item not in event.objects:
                        event.objects.append(obj_item)

            attrs["n_events"] = len(parsed.events)
            attrs["classification"] = parsed.classification
            attrs["requires_temporal"] = parsed.requires_temporal
            attrs["answer_type"] = parsed.answer_type
            if response.usage_metadata:
                attrs["tokens_in"] = response.usage_metadata.get("input_tokens", 0)
                attrs["tokens_out"] = response.usage_metadata.get("output_tokens", 0)

            return parsed

    except Exception as e:
        # LLM failed → fallback
        return _fallback_parse(question, lang, regex_result, trace_id, error=str(e))


async def _fallback_parse(
    question: str,
    lang: str,
    regex_result: dict,
    trace_id: str,
    error: str | None = None,
) -> ParsedQuery:
    """Fallback when LLM is unavailable: translate + regex only."""
    with span(trace_id, "qe.fallback", error=error):
        text_en = question
        if lang == "vi" or (lang == "auto" and _looks_vietnamese(question)):
            try:
                import asyncio
                from ..translator import translate_vi_to_en
                text_en = await asyncio.to_thread(translate_vi_to_en, question)
            except Exception:
                pass

        return ParsedQuery(
            events=[EventQuery(
                text_vi=question if lang != "en" else "",
                text_en=text_en,
                ocr=regex_result.get("ocr", []),
                objects=regex_result.get("objects", []),
            )],
            answer_event=0,
            question=question,
            answer_type="text",
            fallback=True,
        )


def _looks_vietnamese(text: str) -> bool:
    """Simple heuristic: contains Vietnamese-specific characters."""
    vi_chars = set("àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ")
    return any(c in vi_chars for c in text.lower())

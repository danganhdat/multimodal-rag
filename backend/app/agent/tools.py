"""Agent tools: search, look, neighbors, final_answer.

Each tool is a @tool function with Pydantic arg validation.
These are used by the LangGraph ReAct agent.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..config import settings
from ..observe import span

_KF_ID_PATTERN = re.compile(r"^L\d+_V\d+/\d{3,}$")
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_prompt(name: str) -> str:
    path = _PROMPTS_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


class SearchArgs(BaseModel):
    text_en: str = Field(description="English visual scene description to search for")
    k: int = Field(default=20, description="Number of results to return")
    rerank: bool = Field(default=False, description="Apply MLLM reranking on results")


class LookArgs(BaseModel):
    kf_ids: list[str] = Field(description="Keyframe IDs to examine (max 4)", max_length=4)
    question: str = Field(description="What to look for in these frames")


class CropArgs(BaseModel):
    kf_id: str = Field(description="Keyframe ID to crop and inspect")
    bbox: list[float] = Field(description="Normalized crop [x1,y1,x2,y2] in 0..1", min_length=4, max_length=4)
    question: str = Field(description="Detail or text to examine in this crop")


class ReadOCRArgs(BaseModel):
    kf_ids: list[str] = Field(
        description="Keyframe IDs to read OCR text from (max 20)",
        max_length=20,
    )


class NeighborsArgs(BaseModel):
    kf_id: str = Field(description="Keyframe ID to get neighbors of")
    direction: Literal["before", "after", "both"] = Field(
        default="both", description="Direction: before, after, or both"
    )
    n: int = Field(default=5, ge=1, le=20, description="Number of neighbors per direction (1-20)")


class FinalAnswerArgs(BaseModel):
    status: Literal["answered", "uncertain", "not_found", "clarify"] = Field(
        description="Confidence status"
    )
    answer: str = Field(description="The answer text")
    evidence: list[str] = Field(
        default_factory=list,
        description="kf_ids that support this answer (must have been looked at)",
    )


def make_search_tool(milvus, siglip2, bm25, flags=None, trace_id: str = ""):
    @tool("search", args_schema=SearchArgs)
    async def search(text_en: str, k: int = 20, rerank: bool = False) -> str:
        """Search for keyframes matching a visual scene description."""
        from ..pipeline import run_search_multi_event, run_search

        with span(trace_id, "tool.search", query=text_en, k=k, rerank=rerank):
            vec = siglip2.encode_text(text_en)

            if rerank:
                # Use async multi-event path (single event) to get MLLM rerank
                search_flags = dict(flags or {})
                rerank_cfg = dict(search_flags.get("rerank", {}))
                if rerank_cfg.get("method") in (None, "none", "superglobal"):
                    rerank_cfg["method"] = "sg+mllm"
                search_flags["rerank"] = rerank_cfg
                results = await run_search_multi_event(
                    milvus=milvus,
                    siglip2_encoder=siglip2,
                    bm25_index=bm25,
                    events=[{"text_en": text_en, "variants": []}],
                    query_vectors=[vec],
                    flags=search_flags,
                    trace_id=trace_id,
                )
            else:
                results = await run_search(
                    milvus=milvus,
                    siglip2_encoder=siglip2,
                    bm25_index=bm25,
                    text_vectors=[vec],
                    query_text=text_en,
                    flags=flags,
                    trace_id=trace_id,
                )

        top = results[:k]
        if not top:
            return "No results found."

        lines = []
        for i, r in enumerate(top):
            vid = r.get("video_name", "?")
            kidx = r.get("keyframe_idx", 0)
            kf_id = r.get("id", f"{vid}/{kidx:03d}")
            score = r.get("rrf_score", r.get("score", 0))
            objs = ", ".join(r.get("objects", [])[:5])
            ocr = r.get("ocr", "")[:100]
            line = f"{i+1}. {kf_id} (score={score:.3f})"
            if objs:
                line += f" objects=[{objs}]"
            if ocr:
                line += f" ocr=<data>{ocr}</data>"
            lines.append(line)

        return "\n".join(lines)
    return search


class SearchEventsArgs(BaseModel):
    events: list[dict] = Field(description="List of events: [{text_en, variants?}]")
    temporal_relations: list[dict] = Field(
        default_factory=list,
        description="Temporal relations: [{from_event, to_event, type}]",
    )
    k: int = Field(default=20, description="Number of results to return")
    rerank: bool = Field(default=False, description="Apply MLLM reranking on results")


def make_search_events_tool(milvus, siglip2, bm25, flags=None, trace_id: str = ""):
    @tool("search_events", args_schema=SearchEventsArgs)
    async def search_events(
        events: list[dict],
        temporal_relations: list[dict] | None = None,
        k: int = 20,
        rerank: bool = False,
    ) -> str:
        """Multi-event retrieval: per-event search with optional temporal matching and reranking."""
        from ..pipeline import run_search_multi_event

        if not events:
            return "Error: no events provided."

        with span(trace_id, "tool.search_events", n_events=len(events),
                   k=k, rerank=rerank) as attrs:
            query_vectors = []
            for ev in events:
                text = ev.get("text_en", "").strip()
                if not text:
                    return "Error: each event needs text_en."
                query_vectors.append(siglip2.encode_text(text))

            search_flags = dict(flags or {})
            if rerank:
                search_flags.setdefault("rerank", {})["method"] = "mllm"

            if temporal_relations:
                temporal = search_flags.setdefault("temporal", {})
                temporal["enabled"] = True
                temporal["requires_temporal"] = True

            results = await run_search_multi_event(
                milvus=milvus,
                siglip2_encoder=siglip2,
                bm25_index=bm25,
                events=events,
                query_vectors=query_vectors,
                answer_event=0,
                flags=search_flags,
                trace_id=trace_id,
            )

            attrs["n_results"] = len(results)

        top = results[:k]
        if not top:
            return "No results found."

        lines = []
        for i, r in enumerate(top):
            vid = r.get("video_name", "?")
            kidx = r.get("keyframe_idx", 0)
            kf_id = r.get("id", f"{vid}/{kidx:03d}")
            score = r.get("rrf_score", r.get("score", 0))
            objs = ", ".join(r.get("objects", [])[:5])
            ocr = r.get("ocr", "")[:100]
            line = f"{i+1}. {kf_id} (score={score:.3f})"
            if objs:
                line += f" objects=[{objs}]"
            if ocr:
                line += f" ocr=<data>{ocr}</data>"
            lines.append(line)

        return "\n".join(lines)
    return search_events


def make_read_ocr_tool(milvus, trace_id: str = ""):
    @tool("read_ocr", args_schema=ReadOCRArgs)
    def read_ocr(kf_ids: list[str]) -> str:
        """Read OCR text and detected objects for keyframes. Fast index lookup — no VLM call."""
        valid = [k for k in kf_ids[:20] if _KF_ID_PATTERN.match(k)]
        if not valid:
            return "Error: no valid kf_ids. Format: L21_V001/001"

        with span(trace_id, "tool.read_ocr", n_ids=len(valid)):
            rows = milvus.get_by_ids(valid, output_fields=["ocr", "objects"])
            meta = {r["id"]: r for r in rows}

        lines = []
        for kf_id in valid:
            m = meta.get(kf_id, {})
            ocr = m.get("ocr", "")
            objs = ", ".join(m.get("objects", [])[:10])
            line = f"{kf_id}:"
            if ocr:
                line += f" ocr=\"{ocr}\""
            else:
                line += " (no text detected)"
            if objs:
                line += f" objects=[{objs}]"
            lines.append(line)
        return "\n".join(lines)
    return read_ocr


def make_look_tool(milvus, trace_id: str = "", callbacks=None):
    @tool("look", args_schema=LookArgs)
    async def look(kf_ids: list[str], question: str) -> str:
        """Examine keyframe images using VLM to read text, identify objects, etc."""
        from ..llm import get_chat, image_message, LLMUnavailable

        valid_ids = [k for k in kf_ids[:4] if _KF_ID_PATTERN.match(k)]
        if not valid_ids:
            return "Error: no valid kf_ids. Format: L21_V001/001"

        with span(trace_id, "tool.look", kf_ids=valid_ids, question=question[:100]):
            keyframes_dir = Path(settings.KEYFRAMES_DIR)
            image_paths = []
            inspected_ids = []
            for kf_id in valid_ids:
                parts = kf_id.split("/")
                if len(parts) == 2:
                    img = keyframes_dir / parts[0] / f"{parts[1]}.jpg"
                    if img.exists():
                        image_paths.append(str(img))
                        inspected_ids.append(kf_id)

            if not image_paths:
                return f"Error: no images found for {valid_ids}"

            try:
                chat = get_chat("vlm", temperature=0, max_tokens=2000,
                                extra_body={"reasoning": {"max_tokens": 512}}, timeout=90, callbacks=callbacks)
            except LLMUnavailable as e:
                return f"Error: VLM unavailable — {e}"

            prompt_template = _load_prompt("look_v1.txt")
            prompt = prompt_template.replace("{question}", question) if prompt_template else question
            msg = image_message(prompt, image_paths, max_side=768)
            response = await chat.ainvoke([msg])
            return json.dumps({"frame_ids": inspected_ids, "image_paths": image_paths,
                               "observation": response.content}, ensure_ascii=False)
    return look


def make_crop_tool(trace_id: str = "", callbacks=None):
    @tool("crop", args_schema=CropArgs)
    async def crop(kf_id: str, bbox: list[float], question: str) -> str:
        """Crop a keyframe region and inspect its original pixels, useful for small text or objects."""
        from PIL import Image
        from ..llm import get_chat, image_message

        if not _KF_ID_PATTERN.match(kf_id) or len(bbox) != 4 or any(not 0 <= x <= 1 for x in bbox):
            return "Error: invalid frame ID or normalized crop bounds"
        video, n = kf_id.split("/")
        image = Path(settings.KEYFRAMES_DIR) / video / f"{n}.jpg"
        if not image.is_file():
            return "Error: source image missing"
        with Image.open(image) as im:
            w, h = im.size
            box = tuple(round(v * size) for v, size in zip(bbox, (w, h, w, h)))
            if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                return "Error: crop is empty or too small"
            digest = hashlib.sha256(f"{kf_id}:{bbox}:{image.stat().st_mtime_ns}".encode()).hexdigest()[:16]
            dest = Path("data/agent_crops") / f"{video}_{n}_{digest}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            im.convert("RGB").crop(box).save(dest, quality=95)
        with span(trace_id, "tool.crop", kf_id=kf_id, bbox=bbox):
            chat = get_chat("vlm", temperature=0, max_tokens=2000,
                            extra_body={"reasoning": {"max_tokens": 512}}, timeout=90, callbacks=callbacks)
            response = await chat.ainvoke([image_message(question + " Answer strictly from visible crop pixels.", [dest], max_side=1024)])
        return json.dumps({"frame_ids": [kf_id], "image_paths": [str(dest.resolve())],
                           "observation": response.content}, ensure_ascii=False)
    return crop


def make_neighbors_tool(milvus, trace_id: str = ""):
    @tool("neighbors", args_schema=NeighborsArgs)
    def neighbors(kf_id: str, direction: str = "both", n: int = 5) -> str:
        """Get neighboring keyframes (before/after) in the same video."""
        if not _KF_ID_PATTERN.match(kf_id):
            return "Error: invalid kf_id format. Use L21_V001/001"

        with span(trace_id, "tool.neighbors", kf_id=kf_id, direction=direction):
            parts = kf_id.split("/")
            video_name = parts[0]
            kidx = int(parts[1])

            rows = milvus.get_neighbors(video_name, kidx, window=n)
            if not rows:
                return f"No neighbors found for {kf_id}"

            lines = []
            for r in rows:
                nkidx = r.get("keyframe_idx", 0)
                nid = f"{video_name}/{nkidx:03d}"
                pts = r.get("pts_time", 0.0)
                rel = "before" if nkidx < kidx else "after" if nkidx > kidx else "self"
                if direction == "before" and nkidx >= kidx:
                    continue
                if direction == "after" and nkidx <= kidx:
                    continue
                lines.append(f"{nid} ({rel}, {pts:.1f}s)")

            return "\n".join(lines) if lines else f"No {direction} neighbors for {kf_id}"
    return neighbors


@tool("final_answer", args_schema=FinalAnswerArgs)
def final_answer(status: str, answer: str, evidence: list[str] | None = None) -> str:
    """Submit the final answer with evidence."""
    return f"FINAL: status={status} answer={answer} evidence={evidence or []}"

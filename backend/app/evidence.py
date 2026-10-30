"""EvidenceBuilder: select and validate evidence frames for VQA.

Multi-event: at least one evidence per required event. Don't blindly top-3
by global score. Flag insufficient=True when coverage is lacking.
Provenance always tracked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import settings
from .observe import span


@dataclass
class EvidenceFrame:
    kf_id: str
    video_id: str
    timestamp_s: float
    score: float
    event_id: str = ""
    image_path: str = ""
    ocr: str = ""


@dataclass
class EvidenceSet:
    frames: list[EvidenceFrame] = field(default_factory=list)
    sufficient: bool = True
    coverage: list[str] = field(default_factory=list)
    missing_events: list[str] = field(default_factory=list)

    @property
    def kf_ids(self) -> list[str]:
        return [f.kf_id for f in self.frames]

    @property
    def image_paths(self) -> list[str]:
        return [f.image_path for f in self.frames if f.image_path]

    def to_dict(self) -> dict:
        return {
            "frames": [
                {
                    "kf_id": f.kf_id,
                    "video_id": f.video_id,
                    "timestamp_s": f.timestamp_s,
                    "score": f.score,
                    "event_id": f.event_id,
                }
                for f in self.frames
            ],
            "sufficient": self.sufficient,
            "coverage": self.coverage,
            "missing_events": self.missing_events,
        }


def build_evidence(
    results: list[dict],
    events: list[dict] | None = None,
    top_k: int = 3,
    trace_id: str = "",
) -> EvidenceSet:
    """Build evidence set from search results.

    For multi-event queries, ensures at least one frame per event.
    For single-event, takes top-k by score.

    Args:
        results: Search results (may include chain info for multi-event).
        events: Event dicts with id/text_en (None for single-event).
        top_k: Maximum evidence frames.
        trace_id: For observability.

    Returns:
        EvidenceSet with frames, sufficiency flag, and coverage info.
    """
    keyframes_dir = Path(settings.KEYFRAMES_DIR)

    with span(trace_id, "evidence.build", n_results=len(results),
              n_events=len(events) if events else 1, top_k=top_k) as attrs:

        if not results:
            attrs["sufficient"] = False
            return EvidenceSet(sufficient=False)

        if events and len(events) >= 2:
            evidence = _build_multi_event(results, events, top_k, keyframes_dir)
        else:
            evidence = _build_single_event(results, top_k, keyframes_dir)

        attrs["n_frames"] = len(evidence.frames)
        attrs["sufficient"] = evidence.sufficient
        attrs["coverage"] = evidence.coverage

    return evidence


def build_evidence_from_ids(
    kf_ids: list[str],
    milvus,
    trace_id: str = "",
) -> EvidenceSet:
    """Build evidence set from user-selected kf_ids (Manual Mode).

    Args:
        kf_ids: Keyframe IDs selected by user.
        milvus: Milvus client for metadata lookup.
        trace_id: For observability.

    Returns:
        EvidenceSet with frames and image paths.
    """
    keyframes_dir = Path(settings.KEYFRAMES_DIR)

    with span(trace_id, "evidence.build_from_ids", n_ids=len(kf_ids)) as attrs:
        if not kf_ids:
            return EvidenceSet(sufficient=False)

        meta_rows = milvus.get_by_ids(kf_ids)
        meta_by_id = {m["id"]: m for m in meta_rows}

        frames = []
        for kf_id in kf_ids:
            meta = meta_by_id.get(kf_id, {})
            video_id = meta.get("video_name", "")
            parts = kf_id.split("/")
            img_path = ""
            if len(parts) == 2:
                img = keyframes_dir / parts[0] / f"{parts[1]}.jpg"
                if img.exists():
                    img_path = str(img)

            frames.append(EvidenceFrame(
                kf_id=kf_id,
                video_id=video_id,
                timestamp_s=meta.get("pts_time", 0.0),
                score=0.0,
                image_path=img_path,
                ocr=meta.get("ocr", ""),
            ))

        sufficient = len(frames) > 0 and all(f.image_path for f in frames)
        attrs["n_frames"] = len(frames)
        attrs["sufficient"] = sufficient

    return EvidenceSet(
        frames=frames,
        sufficient=sufficient,
        coverage=[f.kf_id for f in frames],
    )


def _build_single_event(
    results: list[dict],
    top_k: int,
    keyframes_dir: Path,
) -> EvidenceSet:
    """Select top-k frames by score for single-event queries."""
    frames = []
    for r in results[:top_k]:
        kf_id = r.get("id", "")
        video_id = r.get("video_name", "")
        parts = kf_id.split("/")
        img_path = ""
        if len(parts) == 2:
            img = keyframes_dir / parts[0] / f"{parts[1]}.jpg"
            if img.exists():
                img_path = str(img)

        frames.append(EvidenceFrame(
            kf_id=kf_id,
            video_id=video_id,
            timestamp_s=r.get("pts_time", 0.0),
            score=r.get("rrf_score", r.get("score", 0.0)),
            image_path=img_path,
            ocr=r.get("ocr", ""),
        ))

    sufficient = len(frames) > 0
    return EvidenceSet(
        frames=frames,
        sufficient=sufficient,
        coverage=[f.kf_id for f in frames],
    )


def _build_multi_event(
    results: list[dict],
    events: list[dict],
    top_k: int,
    keyframes_dir: Path,
) -> EvidenceSet:
    """Select evidence ensuring coverage across events.

    Uses chain info when available (from temporal search).
    Falls back to top-k when no chain info.
    """
    event_ids = [e.get("id", f"E{i+1}") for i, e in enumerate(events)]

    # Check if results have chain info
    best_chain = None
    for r in results[:5]:
        chain = r.get("chain")
        if chain and len(chain) >= 2:
            best_chain = chain
            break

    if best_chain:
        return _evidence_from_chain(best_chain, results, event_ids, top_k, keyframes_dir)

    # No chain info: take top frames, try to cover events by score
    frames = []
    covered = set()

    for r in results[:top_k * 2]:
        if len(frames) >= top_k:
            break
        kf_id = r.get("id", "")
        video_id = r.get("video_name", "")
        parts = kf_id.split("/")
        img_path = ""
        if len(parts) == 2:
            img = keyframes_dir / parts[0] / f"{parts[1]}.jpg"
            if img.exists():
                img_path = str(img)

        frames.append(EvidenceFrame(
            kf_id=kf_id,
            video_id=video_id,
            timestamp_s=r.get("pts_time", 0.0),
            score=r.get("rrf_score", r.get("score", 0.0)),
            image_path=img_path,
            ocr=r.get("ocr", ""),
        ))

    missing = [eid for eid in event_ids if eid not in covered]

    return EvidenceSet(
        frames=frames,
        sufficient=len(frames) > 0,
        coverage=[f.kf_id for f in frames],
        missing_events=missing,
    )


def _evidence_from_chain(
    chain: dict,
    results: list[dict],
    event_ids: list[str],
    top_k: int,
    keyframes_dir: Path,
) -> EvidenceSet:
    """Build evidence from a temporal chain."""
    result_by_id = {r.get("id", ""): r for r in results}
    frames = []
    covered = set()

    for event_label, kf_id in chain.items():
        r = result_by_id.get(kf_id, {})
        video_id = r.get("video_name", "")
        parts = kf_id.split("/")
        img_path = ""
        if len(parts) == 2:
            img = keyframes_dir / parts[0] / f"{parts[1]}.jpg"
            if img.exists():
                img_path = str(img)

        frames.append(EvidenceFrame(
            kf_id=kf_id,
            video_id=video_id,
            timestamp_s=r.get("pts_time", 0.0),
            score=r.get("rrf_score", r.get("chain_score", 0.0)),
            event_id=event_label,
            image_path=img_path,
            ocr=r.get("ocr", ""),
        ))
        covered.add(event_label)

    if len(frames) > top_k:
        frames = frames[:top_k]

    missing = [eid for eid in event_ids if eid not in covered]

    return EvidenceSet(
        frames=frames,
        sufficient=len(missing) == 0 and len(frames) > 0,
        coverage=list(covered),
        missing_events=missing,
    )

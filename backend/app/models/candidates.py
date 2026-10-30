"""Candidate and evidence types with provenance tracking."""

from pydantic import BaseModel


class FrameHit(BaseModel):
    kf_id: str
    video_id: str
    keyframe_idx: int = 0
    frame_idx: int = 0
    timestamp_s: float = 0.0
    event_id: str = ""
    score_dense: float = 0.0
    score_rrf: float = 0.0
    score_boosted: float = 0.0
    score_rerank: float = 0.0
    source: str = ""


class ChainFrame(BaseModel):
    event_id: str
    kf_id: str
    timestamp_s: float = 0.0
    score: float = 0.0


class TemporalChain(BaseModel):
    chain_id: str = ""
    video_id: str
    frames: list[ChainFrame] = []
    chain_score: float = 0.0
    coverage: list[str] = []
    partial: bool = False


class EvidenceItem(BaseModel):
    kf_id: str
    video_id: str
    timestamp_s: float = 0.0
    event_id: str = ""
    source: str = ""


class EvidenceSet(BaseModel):
    items: list[EvidenceItem] = []
    covered_events: list[str] = []
    sufficient: bool = True

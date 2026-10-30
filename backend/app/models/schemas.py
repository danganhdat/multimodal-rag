from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RankerType(str, Enum):
    RRF = "rrf"
    WEIGHTED = "weighted"


class MetricType(str, Enum):
    COSINE = "COSINE"


class QueryItem(BaseModel):
    text: str
    lang: str = "en"


class SearchConfig(BaseModel):
    ranker: RankerType = RankerType.RRF
    rrf_k: int = 60
    weights: list[float] = []
    metric_type: MetricType = MetricType.COSINE
    top_k: int = 50
    sub_k: int = 100
    neighbor_window: int = 5
    group_by_video: bool = False
    use_client_merge: bool = False


class SearchFilters(BaseModel):
    objects: list[str] = []
    ocr: list[str] = []


class SearchRequest(BaseModel):
    queries: list[QueryItem]
    image_query: Optional[str] = None
    config: SearchConfig = Field(default_factory=SearchConfig)
    filters: Optional[SearchFilters] = None


class NeighborFrame(BaseModel):
    keyframe_idx: int
    frame_idx: int
    pts_time: float
    image_url: str


class SearchResult(BaseModel):
    id: str
    video_name: str
    keyframe_idx: int
    frame_idx: int
    pts_time: float
    fps: float
    score: float
    objects: list[str]
    ocr: str = ""
    neighbors: list[NeighborFrame]
    image_url: str


class SearchResponse(BaseModel):
    results: list[SearchResult]
    translations: dict[str, str] = {}
    search_time_ms: float


class TranslateRequest(BaseModel):
    text: str
    source: str = "vi"
    target: str = "en"


class TranslateResponse(BaseModel):
    original: str
    translated: str

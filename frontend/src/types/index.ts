export interface QueryItem {
  text: string;
  lang: "en" | "vi";
}

export type RankerType = "rrf" | "weighted";
export type MetricType = "COSINE";

export interface SearchConfig {
  ranker: RankerType;
  rrf_k: number;
  weights: number[];
  metric_type: MetricType;
  top_k: number;
  sub_k: number;
  neighbor_window: number;
  group_by_video: boolean;
  use_client_merge: boolean;
}

export interface SearchFilters {
  objects: string[];
  ocr: string[];
}

export interface SearchRequest {
  queries: QueryItem[];
  image_query?: string;
  config: SearchConfig;
  filters?: SearchFilters;
}

export interface NeighborFrame {
  keyframe_idx: number;
  frame_idx: number;
  pts_time: number;
  image_url: string;
}

export interface SearchResult {
  id: string;
  video_name: string;
  keyframe_idx: number;
  frame_idx: number;
  pts_time: number;
  fps: number;
  score: number;
  objects: string[];
  ocr: string;
  neighbors: NeighborFrame[];
  image_url: string;
}

export interface SearchResponse {
  results: SearchResult[];
  translations: Record<string, string>;
  search_time_ms: number;
}

export interface TranslateResponse {
  original: string;
  translated: string;
}

export interface VideoMeta {
  title?: string;
  description?: string;
  author?: string;
  publish_date?: string;
  watch_url?: string;
  length?: number;
  keywords?: string[];
}

export const DEFAULT_CONFIG: SearchConfig = {
  ranker: "rrf",
  rrf_k: 60,
  weights: [],
  metric_type: "COSINE",
  top_k: 50,
  sub_k: 100,
  neighbor_window: 5,
  group_by_video: false,
  use_client_merge: false,
};

// ── v1 types ──

export interface EventQuery {
  id: string;
  text_vi: string;
  text_en: string;
  ocr: string[];
  objects: string[];
  variants: string[];
  enabled: boolean;
}

export interface TemporalRelation {
  from_event: string;
  to_event: string;
  type: "before" | "after";
}

export interface ParsedQuery {
  classification: "single_scene" | "multi_event";
  events: EventQuery[];
  temporal_relations: TemporalRelation[];
  requires_temporal: boolean;
  answer_event: number;
  question: string;
  answer_type: string;
  variants: string[];
  fallback: boolean;
}

// ── Candidate / Evidence types ──

export interface FrameHit {
  kf_id: string;
  video_id: string;
  keyframe_idx: number;
  frame_idx: number;
  timestamp_s: number;
  event_id: string;
  score_dense: number;
  score_rrf: number;
  score_boosted: number;
  score_rerank: number;
  source: string;
}

export interface ChainFrame {
  event_id: string;
  kf_id: string;
  timestamp_s: number;
  score: number;
}

export interface TemporalChain {
  chain_id: string;
  video_id: string;
  frames: ChainFrame[];
  chain_score: number;
  coverage: string[];
  partial: boolean;
}

export interface EvidenceItem {
  kf_id: string;
  video_id: string;
  timestamp_s: number;
  event_id: string;
  source: string;
}

export interface EvidenceSet {
  items: EvidenceItem[];
  covered_events: string[];
  sufficient: boolean;
}

// ── VQA types ──

export interface VQAAnswerRequest {
  question: string;
  evidence_ids: string[];
  retrieval_trace_id?: string;
}

export interface VQARequest {
  query: string;
  k?: number;
  flags?: Record<string, unknown>;
}

export interface VQAResponse {
  status: "answered" | "insufficient_evidence" | "not_implemented" | "error";
  answer: string;
  evidence: string[];
  trace_id: string;
  search_time_ms?: number;
  answer_time_ms?: number;
  usage: Record<string, unknown>;
}

export interface ParseResponse {
  parsed: ParsedQuery;
  regex: { ocr: string[]; objects: string[] };
  trace_id: string;
}

export interface V1SearchRequest {
  question?: string;
  parsed?: ParsedQuery;
  k?: number;
  upvote_kf_ids?: string[];
  downvote_kf_ids?: string[];
  flags?: Record<string, unknown>;
}

export interface V1SearchResult {
  id: string;
  video_name: string;
  keyframe_idx: number;
  frame_idx: number;
  pts_time: number;
  fps: number;
  score: number;
  objects: string[];
  ocr: string;
  sources: Record<string, number>;
  image_url: string;
  neighbors: NeighborFrame[];
}

export interface V1SearchResponse {
  results: V1SearchResult[];
  trace_id: string;
  latency_ms: number;
}

export interface V1ConfigResponse {
  flags: Record<string, unknown>;
  status: {
    bm25: boolean;
    bm25_n_docs: number;
    llm: boolean;
    vlm: boolean;
    milvus: boolean;
    llm_model: string | null;
    vlm_model: string | null;
    index_version: string;
    encoder: string;
    collection: string;
  };
  config_id: string;
}

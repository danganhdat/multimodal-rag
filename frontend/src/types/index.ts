export interface QueryItem {
  text: string;
  lang: "en" | "vi";
}

export type RankerType = "rrf" | "weighted";
export type MetricType = "COSINE";
export type EncoderMode = "clip" | "siglip2";

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
  rerank: boolean;
  rerank_candidates: number;
  encoder_mode: EncoderMode;
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
  rerank_score: number | null;
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
  rerank: true,
  rerank_candidates: 20,
  encoder_mode: "siglip2",
};

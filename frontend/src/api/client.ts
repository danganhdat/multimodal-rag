import type {
  SearchRequest,
  SearchResponse,
  TranslateResponse,
  VideoMeta,
  ParseResponse,
  V1SearchRequest,
  V1SearchResponse,
  V1ConfigResponse,
  VQAAnswerRequest,
  VQARequest,
  VQAResponse,
} from "../types";

const API_BASE = "/api";
const V1_BASE = "/api/v1";

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export function search(req: SearchRequest): Promise<SearchResponse> {
  return post("/search", req);
}

export function translate(
  text: string,
  source = "vi",
  target = "en"
): Promise<TranslateResponse> {
  return post("/translate", { text, source, target });
}

export function getVideoUrl(videoName: string): string {
  return `${API_BASE}/video/${videoName}`;
}

export function getVideoMeta(videoName: string): Promise<VideoMeta> {
  return get(`/video-meta/${videoName}`);
}

export function getObjectClasses(): Promise<{ classes: string[] }> {
  return get("/object-classes");
}

// ── v1 API ──

async function v1Post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${V1_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    const msg = data?.error?.message || `${res.status} ${res.statusText}`;
    throw new Error(msg);
  }
  return res.json();
}

async function v1Get<T>(path: string): Promise<T> {
  const res = await fetch(`${V1_BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export function parseQuery(
  question: string,
  lang = "auto",
  flags?: Record<string, unknown>
): Promise<ParseResponse> {
  return v1Post("/query/parse", { question, lang, flags });
}

export function v1Search(req: V1SearchRequest): Promise<V1SearchResponse> {
  return v1Post("/search", req);
}

export function v1SearchImage(
  kf_id?: string,
  k?: number
): Promise<V1SearchResponse> {
  return v1Post("/search/image", { kf_id, k });
}

export function v1Config(): Promise<V1ConfigResponse> {
  return v1Get("/config");
}

export function v1VQAAnswer(req: VQAAnswerRequest): Promise<VQAResponse> {
  return v1Post("/vqa/answer", req);
}

export function v1VQA(req: VQARequest): Promise<VQAResponse> {
  return v1Post("/vqa", req);
}

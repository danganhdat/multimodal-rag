import type {
  SearchRequest,
  SearchResponse,
  TranslateResponse,
  VideoMeta,
} from "../types";

const API_BASE = "/api";

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

export function switchEncoder(mode: string): Promise<{ mode: string; status: string }> {
  return post("/switch-encoder?mode=" + encodeURIComponent(mode), {});
}

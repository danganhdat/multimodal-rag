"""
Ground truth annotation tool — search-assisted with anti-contamination.

Flow:
  1. Select a query from sidebar
  2. Click "Search" to get top-K candidates from the CLIP search API
  3. Pick the correct frame from search results (or browse manually)
  4. Rewrite the query in your own words (eval_query) to avoid contamination
  5. Evaluation uses eval_query, not the original query

Usage:
    python -m backend.scripts.create_gt

Requires backend server running on port 8000 (for search API).
Open http://localhost:8501 in browser.
"""

import json
import re
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings

QUERY_DIR = Path(__file__).resolve().parent.parent.parent / "query-p1-groupA"
GT_PATH = Path(__file__).resolve().parent.parent.parent / "eval" / "ground_truth.json"
KEYFRAMES_DIR = settings.KEYFRAMES_DIR
PORT = 8501
API_BASE = "http://localhost:8000/api"


def load_queries():
    queries = []
    for f in sorted(QUERY_DIR.glob("*.txt")):
        if "-trake" in f.name:
            continue
        m = re.match(r"query-p1-(\d+)-(kis|qa)\.txt", f.name)
        if not m:
            continue
        queries.append({
            "id": f.stem,
            "num": int(m.group(1)),
            "type": m.group(2).upper(),
            "text": f.read_text(encoding="utf-8").strip(),
        })
    return sorted(queries, key=lambda q: q["num"])


def list_videos():
    if not KEYFRAMES_DIR.exists():
        return []
    return sorted([d.name for d in KEYFRAMES_DIR.iterdir() if d.is_dir()])


def list_keyframes(video_name):
    vdir = KEYFRAMES_DIR / video_name
    if not vdir.exists():
        return []
    frames = []
    for f in sorted(vdir.glob("*.jpg")):
        m = re.match(r"(\d+)\.jpg", f.name)
        if m:
            frames.append(int(m.group(1)))
    return frames


def load_gt():
    if GT_PATH.exists():
        return json.loads(GT_PATH.read_text(encoding="utf-8"))
    return []


def save_gt(data):
    GT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


QUERIES = load_queries()
VIDEOS = list_videos()


def build_html():
    gt = load_gt()
    gt_map = {g["query_id"]: g for g in gt}

    queries_js = json.dumps(QUERIES, ensure_ascii=False)
    videos_js = json.dumps(VIDEOS)
    gt_js = json.dumps(gt_map, ensure_ascii=False)

    return """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>GT Annotation Tool</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui; background: #f5f5f5; display: flex; height: 100vh; }
  .sidebar { width: 380px; background: #fff; border-right: 1px solid #ddd; overflow-y: auto; padding: 12px; flex-shrink: 0; }
  .main { flex: 1; overflow-y: auto; padding: 16px; }
  .query-item { padding: 8px; margin: 4px 0; border-radius: 6px; cursor: pointer; border: 2px solid transparent; font-size: 13px; }
  .query-item:hover { background: #e3f2fd; }
  .query-item.active { border-color: #1976d2; background: #e3f2fd; }
  .query-item.done { border-color: #4caf50; background: #e8f5e9; }
  .query-item .tag { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: bold; margin-right: 6px; }
  .tag-kis { background: #bbdefb; color: #1565c0; }
  .tag-qa { background: #ffe0b2; color: #e65100; }
  .query-text { color: #333; margin-top: 4px; font-size: 12px; white-space: pre-wrap; }

  h2 { font-size: 16px; margin-bottom: 12px; }
  h3 { font-size: 14px; margin: 8px 0; }
  .section { background: #fff; border-radius: 8px; padding: 12px; margin-bottom: 12px; border: 1px solid #e0e0e0; }
  .section-title { font-size: 14px; font-weight: bold; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }

  .video-select { width: 100%; padding: 8px; font-size: 14px; margin-bottom: 8px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 6px; }
  .frame { cursor: pointer; border: 3px solid transparent; border-radius: 4px; position: relative; }
  .frame:hover { border-color: #1976d2; }
  .frame.selected { border-color: #4caf50; }
  .frame img { width: 100%; display: block; border-radius: 2px; }
  .frame .label { position: absolute; bottom: 2px; right: 4px; background: rgba(0,0,0,.7); color: #fff; font-size: 10px; padding: 1px 4px; border-radius: 2px; }
  .frame .score-label { position: absolute; top: 2px; left: 4px; background: rgba(25,118,210,.85); color: #fff; font-size: 10px; padding: 1px 4px; border-radius: 2px; }

  .status { padding: 8px; background: #fff3e0; border-radius: 6px; margin-bottom: 12px; font-size: 13px; }
  .btn { padding: 6px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; }
  .btn-search { background: #1976d2; color: #fff; }
  .btn-search:disabled { background: #90caf9; cursor: not-allowed; }
  .btn-save { background: #4caf50; color: #fff; }
  .btn-clear { background: #f44336; color: #fff; margin-left: 8px; }
  .progress { font-size: 12px; color: #666; margin-bottom: 8px; }

  .rewrite-section { margin-top: 8px; }
  .rewrite-section textarea { width: 100%; min-height: 60px; padding: 8px; font-size: 13px; border: 1px solid #ccc; border-radius: 4px; resize: vertical; font-family: inherit; }
  .rewrite-section .hint { font-size: 11px; color: #888; margin-top: 4px; }
  .selected-info { background: #e8f5e9; padding: 8px; border-radius: 4px; margin-bottom: 8px; font-size: 13px; display: none; }
  .selected-info img { max-width: 200px; border-radius: 4px; margin-top: 4px; }
</style>
</head><body>

<div class="sidebar">
  <h2>Queries (18 KIS + 3 QA)</h2>
  <div class="progress" id="progress"></div>
  <div id="query-list"></div>
</div>

<div class="main">
  <div class="status" id="status">Chon query ben trai de bat dau.</div>
  <div id="controls" style="display:none">
    <h3 id="current-query"></h3>

    <div class="selected-info" id="selected-info"></div>

    <div class="section">
      <div class="section-title">
        Search Results
        <button class="btn btn-search" id="search-btn" onclick="doSearch()">Search</button>
        <span id="search-status" style="font-size:12px;color:#888"></span>
      </div>
      <div class="grid" id="search-grid"></div>
    </div>

    <div class="section">
      <div class="section-title">Manual Browse</div>
      <select class="video-select" id="video-select" onchange="loadFrames()">
        <option value="">-- Chon video --</option>
      </select>
      <div class="grid" id="frame-grid"></div>
    </div>

    <div class="section rewrite-section">
      <div class="section-title">Rewrite Query (for eval)</div>
      <textarea id="eval-query" placeholder="Viet lai mo ta bang loi khac (dung cho evaluation, tranh contamination)..."></textarea>
      <div class="hint">Eval se dung query nay thay vi query goc. Viet lai sau khi da thay frame dung.</div>
      <button class="btn btn-save" onclick="saveEvalQuery()" style="margin-top:6px">Save Rewrite</button>
    </div>
  </div>
</div>

<script>
const QUERIES = """ + json.dumps(QUERIES, ensure_ascii=False) + """;
const VIDEOS = """ + json.dumps(VIDEOS) + """;
let gt = """ + json.dumps(gt_map, ensure_ascii=False) + """;
let activeQuery = null;

function render() {
  const list = document.getElementById('query-list');
  const done = Object.keys(gt).length;
  document.getElementById('progress').textContent = done + '/' + QUERIES.length + ' annotated';

  list.innerHTML = QUERIES.map(q => {
    const isDone = gt[q.id] ? 'done' : '';
    const isActive = activeQuery && activeQuery.id === q.id ? 'active' : '';
    const gtInfo = gt[q.id] ? ' done ' + gt[q.id].video_name + '/' + gt[q.id].frame_idx : '';
    const hasRewrite = gt[q.id] && gt[q.id].eval_query ? ' [R]' : '';
    return '<div class="query-item ' + isDone + ' ' + isActive + '" onclick="selectQuery(\\'' + q.id + '\\')">' +
      '<span class="tag tag-' + q.type.toLowerCase() + '">' + q.type + '</span>' +
      '<strong>' + q.id + '</strong>' + gtInfo + hasRewrite +
      '<div class="query-text">' + escapeHtml(q.text.substring(0, 200)) + (q.text.length > 200 ? '...' : '') + '</div>' +
      '</div>';
  }).join('');
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function selectQuery(id) {
  activeQuery = QUERIES.find(q => q.id === id);
  document.getElementById('controls').style.display = 'block';
  document.getElementById('current-query').textContent = activeQuery.type + ': ' + activeQuery.text;
  document.getElementById('search-grid').innerHTML = '';
  document.getElementById('search-status').textContent = '';

  // restore eval_query
  const textarea = document.getElementById('eval-query');
  textarea.value = gt[id] && gt[id].eval_query ? gt[id].eval_query : '';

  updateSelectedInfo();

  const sel = document.getElementById('video-select');
  sel.innerHTML = '<option value="">-- Chon video --</option>' +
    VIDEOS.map(v => '<option value="' + v + '"' + (gt[id] && gt[id].video_name === v ? ' selected' : '') + '>' + v + '</option>').join('');

  if (gt[id]) loadFrames();
  else document.getElementById('frame-grid').innerHTML = '';

  document.getElementById('status').innerHTML =
    '<strong>' + activeQuery.id + '</strong> — Click Search de tim candidates, hoac chon video de browse thu cong.' +
    (gt[id] ? ' <button class="btn btn-clear" onclick="clearGt(\\'' + id + '\\')">Xoa GT</button>' : '');

  render();
}

function updateSelectedInfo() {
  const el = document.getElementById('selected-info');
  if (!activeQuery || !gt[activeQuery.id]) {
    el.style.display = 'none';
    return;
  }
  const g = gt[activeQuery.id];
  const pad = String(g.frame_idx).padStart(3, '0');
  el.style.display = 'block';
  el.innerHTML = '<strong>Selected GT:</strong> ' + g.video_name + '/' + pad +
    (g.eval_query ? '<br><strong>Eval query:</strong> ' + escapeHtml(g.eval_query) : '') +
    '<br><img src="/keyframes/' + g.video_name + '/' + pad + '.jpg">';
}

function doSearch() {
  if (!activeQuery) return;
  const btn = document.getElementById('search-btn');
  const statusEl = document.getElementById('search-status');
  btn.disabled = true;
  statusEl.textContent = 'Searching...';

  fetch('/api/search-proxy', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query: activeQuery.text, top_k: 50})
  })
  .then(r => r.json())
  .then(data => {
    btn.disabled = false;
    if (data.error) {
      statusEl.textContent = 'Error: ' + data.error;
      return;
    }
    statusEl.textContent = data.results.length + ' results';
    const grid = document.getElementById('search-grid');
    const selectedId = gt[activeQuery.id];
    grid.innerHTML = data.results.map((r, i) => {
      const pad = String(r.keyframe_idx).padStart(3, '0');
      const sel = selectedId && selectedId.video_name === r.video_name && selectedId.frame_idx === r.keyframe_idx ? 'selected' : '';
      return '<div class="frame ' + sel + '" onclick="pickFrame(\\'' + r.video_name + '\\',' + r.keyframe_idx + ',' + r.frame_idx + ')">' +
        '<img src="/keyframes/' + r.video_name + '/' + pad + '.jpg" loading="lazy">' +
        '<span class="score-label">' + r.score.toFixed(3) + '</span>' +
        '<span class="label">' + r.video_name + '/' + pad + '</span></div>';
    }).join('');
  })
  .catch(e => {
    btn.disabled = false;
    statusEl.textContent = 'Error: ' + e.message + ' (is backend running on port 8000?)';
  });
}

function loadFrames() {
  const video = document.getElementById('video-select').value;
  const grid = document.getElementById('frame-grid');
  if (!video) { grid.innerHTML = ''; return; }

  grid.innerHTML = '<p>Loading...</p>';
  fetch('/api/frames?video=' + video)
    .then(r => r.json())
    .then(frames => {
      const selectedIdx = gt[activeQuery.id] ? gt[activeQuery.id].frame_idx : null;
      const selectedVideo = gt[activeQuery.id] ? gt[activeQuery.id].video_name : null;
      grid.innerHTML = frames.map(idx => {
        const pad = String(idx).padStart(3, '0');
        const sel = (video === selectedVideo && idx === selectedIdx) ? 'selected' : '';
        return '<div class="frame ' + sel + '" onclick="pickFrameManual(\\'' + video + '\\',' + idx + ')">' +
          '<img src="/keyframes/' + video + '/' + pad + '.jpg" loading="lazy">' +
          '<span class="label">' + pad + '</span></div>';
      }).join('');
    });
}

function pickFrame(video, keyframeIdx, frameIdx) {
  if (!activeQuery) return;
  const entry = {
    query_id: activeQuery.id,
    query: activeQuery.text,
    query_type: activeQuery.type,
    video_name: video,
    frame_idx: keyframeIdx,
    frame_range: [Math.max(0, keyframeIdx - 30), keyframeIdx + 30]
  };
  if (gt[activeQuery.id] && gt[activeQuery.id].eval_query) {
    entry.eval_query = gt[activeQuery.id].eval_query;
  }
  gt[activeQuery.id] = entry;
  persistGt();
  doSearch();
  loadFrames();
  updateSelectedInfo();
  render();
}

function pickFrameManual(video, keyframeIdx) {
  if (!activeQuery) return;
  const entry = {
    query_id: activeQuery.id,
    query: activeQuery.text,
    query_type: activeQuery.type,
    video_name: video,
    frame_idx: keyframeIdx,
    frame_range: [Math.max(0, keyframeIdx - 30), keyframeIdx + 30]
  };
  if (gt[activeQuery.id] && gt[activeQuery.id].eval_query) {
    entry.eval_query = gt[activeQuery.id].eval_query;
  }
  gt[activeQuery.id] = entry;
  persistGt();
  loadFrames();
  updateSelectedInfo();
  render();
}

function saveEvalQuery() {
  if (!activeQuery || !gt[activeQuery.id]) {
    alert('Chon frame GT truoc khi viet eval query.');
    return;
  }
  const evalQuery = document.getElementById('eval-query').value.trim();
  if (!evalQuery) {
    alert('Vui long nhap eval query.');
    return;
  }
  gt[activeQuery.id].eval_query = evalQuery;
  persistGt();
  updateSelectedInfo();
  render();
  document.getElementById('status').innerHTML = '<strong>Saved eval_query</strong> for ' + activeQuery.id;
}

function clearGt(id) {
  delete gt[id];
  persistGt();
  render();
  document.getElementById('eval-query').value = '';
  updateSelectedInfo();
  if (activeQuery && activeQuery.id === id) {
    loadFrames();
    document.getElementById('search-grid').innerHTML = '';
  }
}

function persistGt() {
  fetch('/api/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(Object.values(gt))
  });
}

render();
</script>
</body></html>"""


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "":
            html = build_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        elif parsed.path == "/api/frames":
            qs = parse_qs(parsed.query)
            video = qs.get("video", [""])[0]
            frames = list_keyframes(video)
            data = json.dumps(frames).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

        elif parsed.path.startswith("/keyframes/"):
            rel = parsed.path[len("/keyframes/"):]
            fpath = KEYFRAMES_DIR / rel
            if fpath.exists() and fpath.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(fpath.read_bytes())
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/save":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            save_gt(body)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        elif parsed.path == "/api/search-proxy":
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length))
            try:
                resp = requests.post(
                    f"{API_BASE}/search",
                    json={
                        "queries": [{"text": req["query"], "lang": "vi"}],
                        "config": {"top_k": req.get("top_k", 50), "rerank": False},
                    },
                    timeout=30,
                )
                data = resp.json()
                results = [
                    {
                        "video_name": r["video_name"],
                        "keyframe_idx": r["keyframe_idx"],
                        "frame_idx": r.get("frame_idx", 0),
                        "score": r["score"],
                    }
                    for r in data.get("results", [])
                ]
                out = json.dumps({"results": results}).encode()
            except Exception as e:
                out = json.dumps({"error": str(e)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    print(f"Queries: {len(QUERIES)} (18 KIS + 3 QA)")
    print(f"Videos: {len(VIDEOS)}")
    print(f"Keyframes dir: {KEYFRAMES_DIR}")
    print(f"GT output: {GT_PATH}")
    print(f"Search API: {API_BASE}")
    print(f"\nOpen http://localhost:{PORT}")
    server = HTTPServer(("", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")

"""
Pre-fill ground truth candidates by bulk-searching all queries.

Splits each query paragraph into sentences, sends as multi-query (RRF fusion),
and saves top candidates per query for manual review.

Usage:
    python -m backend.scripts.prefill_gt
    python -m backend.scripts.prefill_gt --top-k 10 --output eval/gt_candidates.json
"""

import argparse
import io
import json
import re
import sys
from pathlib import Path

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

API_BASE = "http://localhost:8000/api"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUERY_DIR = PROJECT_ROOT / "query-p1-groupA"
GT_PATH = PROJECT_ROOT / "eval" / "ground_truth.json"


def split_sentences_vi(text: str) -> list[str]:
    """Split Vietnamese paragraph into sentences at period/newline boundaries."""
    parts = re.split(r"(?<=\.)\s+|\n+", text.strip())
    sentences = [s.strip() for s in parts if len(s.strip()) >= 10]
    return sentences if sentences else [text.strip()]


def load_queries() -> list[dict]:
    """Load query files, skip TRAKE queries."""
    queries = []
    for f in sorted(QUERY_DIR.glob("query-p1-*-*.txt")):
        if "trake" in f.stem.lower():
            continue
        text = f.read_text(encoding="utf-8").strip()
        parts = f.stem.split("-")
        query_num = parts[2]
        query_type = parts[3].upper()
        queries.append({
            "query_id": f.stem,
            "query_num": int(query_num),
            "query_type": query_type,
            "query": text,
        })
    queries.sort(key=lambda q: q["query_num"])
    return queries


def search_multi(sentences: list[str], lang: str = "vi", top_k: int = 10) -> dict:
    """Search with sentence-split multi-query + RRF + rerank."""
    body = {
        "queries": [{"text": s, "lang": lang} for s in sentences],
        "config": {
            "top_k": top_k,
            "rerank": True,
            "rerank_candidates": 200,
            "encoder_mode": "siglip2",
        },
    }
    resp = requests.post(f"{API_BASE}/search", json=body, timeout=120)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Pre-fill GT candidates")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "eval" / "gt_candidates.json",
    )
    parser.add_argument("--resume", action="store_true", help="Resume: skip queries with existing candidates")
    args = parser.parse_args()

    if not QUERY_DIR.exists():
        print(f"Query directory not found: {QUERY_DIR}")
        sys.exit(1)

    queries = load_queries()
    print(f"Loaded {len(queries)} queries (skipping TRAKE)")

    existing = {}
    if args.resume and args.output.exists():
        prev = json.loads(args.output.read_text(encoding="utf-8"))
        existing = {e["query_id"]: e for e in prev if e.get("candidates")}
        print(f"Resuming: {len(existing)} queries already have candidates")

    results = []
    for i, q in enumerate(queries):
        if q["query_id"] in existing:
            print(f"\n[{i + 1}/{len(queries)}] {q['query_id']} — SKIP (already has {len(existing[q['query_id']]['candidates'])} candidates)")
            results.append(existing[q["query_id"]])
            continue

        sentences = split_sentences_vi(q["query"])
        print(f"\n[{i + 1}/{len(queries)}] {q['query_id']} ({q['query_type']})")
        print(f"  Sentences ({len(sentences)}):")
        for j, s in enumerate(sentences):
            print(f"    {j + 1}. {s[:80]}{'...' if len(s) > 80 else ''}")

        try:
            resp = search_multi(sentences, top_k=args.top_k)
            candidates = []
            for rank, r in enumerate(resp["results"], 1):
                candidates.append({
                    "rank": rank,
                    "video_name": r["video_name"],
                    "keyframe_idx": r["keyframe_idx"],
                    "frame_idx": r["frame_idx"],
                    "pts_time": round(r["pts_time"], 2),
                    "score": round(r["score"], 4),
                    "rerank_score": round(r["rerank_score"], 4) if r.get("rerank_score") is not None else None,
                    "objects": r.get("objects", [])[:5],
                    "ocr": r.get("ocr", "")[:100],
                })
            print(f"  Top-1: {candidates[0]['video_name']} frame={candidates[0]['frame_idx']} score={candidates[0]['score']}" if candidates else "  No results")
        except Exception as e:
            print(f"  ERROR: {e}")
            candidates = []

        results.append({
            "query_id": q["query_id"],
            "query_type": q["query_type"],
            "query": q["query"],
            "sentences": sentences,
            "num_sentences": len(sentences),
            "candidates": candidates,
            # Fields for manual annotation:
            "selected": None,
            "video_name": None,
            "frame_idx": None,
            "frame_range": None,
            "eval_query": None,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n{'=' * 60}")
    print(f"Saved {len(results)} entries to {args.output}")
    print(f"Next: Review candidates, fill 'selected', 'video_name', 'frame_idx', 'frame_range', 'eval_query'")
    print(f"Then convert to {GT_PATH} for evaluation")


if __name__ == "__main__":
    main()

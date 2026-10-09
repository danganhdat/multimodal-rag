"""
Evaluation pipeline for AIC25 multimodal search.

Always splits query paragraphs into sentences and uses multi-query RRF fusion.
Primary mode: SigLIP2 + Rerank.

Metrics:
  - Mean of Top-k R-Scores (competition metric)
  - Recall@k
  - MRR (Mean Reciprocal Rank)

Usage:
    python -m backend.scripts.evaluate --ground-truth eval/ground_truth.json
    python -m backend.scripts.evaluate --ground-truth eval/ground_truth.json --no-rerank
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

API_BASE = "http://localhost:8000/api"
K_VALUES = [1, 5, 20, 50, 100]


def split_sentences_vi(text: str) -> list[str]:
    """Split Vietnamese paragraph into sentences at period/newline boundaries."""
    parts = re.split(r"(?<=\.)\s+|\n+", text.strip())
    sentences = [s.strip() for s in parts if len(s.strip()) >= 10]
    return sentences if sentences else [text.strip()]


def load_ground_truth(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data:
        if "frame_range" not in item:
            fi = item["frame_idx"]
            item["frame_range"] = [fi - 30, fi + 30]
    return data


def search(
    query: str,
    lang: str = "vi",
    rerank: bool = True,
    top_k: int = 100,
    rerank_candidates: int = 200,
) -> tuple[list[dict], float, int]:
    """Search with sentence-split multi-query + RRF.

    Returns (results, latency_ms, num_sentences).
    """
    sentences = split_sentences_vi(query)
    body = {
        "queries": [{"text": s, "lang": lang} for s in sentences],
        "config": {
            "top_k": top_k,
            "rerank": rerank,
            "rerank_candidates": rerank_candidates,
            "encoder_mode": "siglip2",
        },
    }
    resp = requests.post(f"{API_BASE}/search", json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["results"], data.get("search_time_ms", 0), len(sentences)


def r_score_kis(result: dict, gt: dict) -> float:
    if result["video_name"] != gt["video_name"]:
        return 0.0
    s, e = gt["frame_range"]
    if s <= result["frame_idx"] <= e:
        return 1.0
    return 0.0


def mean_topk_rscore(results: list[dict], gt: dict) -> float:
    total = 0.0
    for k in K_VALUES:
        max_r = 0.0
        for i in range(min(k, len(results))):
            r = r_score_kis(results[i], gt)
            if r > max_r:
                max_r = r
        total += max_r
    return total / len(K_VALUES)


def recall_at_k(results: list[dict], gt: dict, k: int) -> float:
    for i in range(min(k, len(results))):
        if r_score_kis(results[i], gt) > 0:
            return 1.0
    return 0.0


def mrr(results: list[dict], gt: dict) -> float:
    for i, r in enumerate(results):
        if r_score_kis(r, gt) > 0:
            return 1.0 / (i + 1)
    return 0.0


def first_hit_rank(results: list[dict], gt: dict) -> int:
    """Return 1-indexed rank of first correct result, or -1 if not found."""
    for i, r in enumerate(results):
        if r_score_kis(r, gt) > 0:
            return i + 1
    return -1


def evaluate(
    ground_truth: list[dict],
    rerank: bool = True,
    rerank_candidates: int = 200,
) -> dict:
    scores = []
    recalls = {k: [] for k in K_VALUES}
    mrr_scores = []
    per_query = []
    total_sentences = 0

    for i, gt in enumerate(ground_truth):
        query = gt.get("eval_query") or gt["query"]
        query_id = gt.get("query_id", f"q{i + 1}")
        query_type = gt.get("query_type", "KIS")

        print(f"  [{i + 1}/{len(ground_truth)}] {query_id} ({query_type})", end=" ")

        lang = gt.get("lang", "vi")
        try:
            results, latency, n_sentences = search(
                query, lang=lang, rerank=rerank, rerank_candidates=rerank_candidates
            )
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        total_sentences += n_sentences
        s = mean_topk_rscore(results, gt)
        scores.append(s)

        for k in K_VALUES:
            recalls[k].append(recall_at_k(results, gt, k))

        m = mrr(results, gt)
        mrr_scores.append(m)
        hr = first_hit_rank(results, gt)

        status = f"HIT@{hr}" if hr > 0 else "MISS"
        print(f"[{n_sentences} sent] score={s:.3f} {status} latency={latency:.0f}ms")

        per_query.append({
            "query_id": query_id,
            "query_type": query_type,
            "query": query[:120],
            "num_sentences": n_sentences,
            "score": round(s, 4),
            "mrr": round(m, 4),
            "recalls": {str(k): round(recall_at_k(results, gt, k), 1) for k in K_VALUES},
            "latency_ms": round(latency, 1),
            "hit_rank": hr,
            "top1_video": results[0]["video_name"] if results else "",
            "top1_frame": results[0]["frame_idx"] if results else -1,
            "top1_score": round(results[0]["score"], 4) if results else 0,
        })

    n = len(scores) or 1
    return {
        "summary": {
            "mean_topk_rscore": round(sum(scores) / n, 4),
            "mrr": round(sum(mrr_scores) / n, 4),
            **{f"recall@{k}": round(sum(recalls[k]) / n, 4) for k in K_VALUES},
            "num_queries": len(scores),
            "avg_sentences": round(total_sentences / n, 1),
        },
        "per_query": per_query,
    }


def print_summary(summary: dict) -> None:
    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Queries:          {summary['num_queries']}")
    print(f"  Avg sentences:    {summary['avg_sentences']}")
    print(f"  Mean TopK-R:      {summary['mean_topk_rscore']:.4f}")
    print(f"  MRR:              {summary['mrr']:.4f}")
    for k in K_VALUES:
        print(f"  Recall@{k:<3}:       {summary[f'recall@{k}']:.4f}")
    print("=" * 70)


def print_per_query_table(per_query: list[dict]) -> None:
    print()
    headers = ["ID", "Type", "Sent", "Score", "MRR", "Hit@", "Top-1 Video", "Latency"]
    widths = [20, 4, 4, 6, 6, 5, 12, 8]

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep_line = "-+-".join("-" * w for w in widths)

    print(header_line)
    print(sep_line)
    for q in per_query:
        hr = str(q["hit_rank"]) if q["hit_rank"] > 0 else "-"
        row = [
            q["query_id"][:20],
            q["query_type"][:4],
            str(q["num_sentences"]),
            f"{q['score']:.4f}",
            f"{q['mrr']:.4f}",
            hr,
            q["top1_video"][:12],
            f"{q['latency_ms']:.0f}ms",
        ]
        print(" | ".join(c.ljust(w) for c, w in zip(row, widths)))
    print()


def main():
    parser = argparse.ArgumentParser(description="AIC25 Search Evaluation")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("eval/ground_truth.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/results.json"),
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable reranking (dense-only evaluation)",
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=200,
        help="Number of tier-1 candidates for reranker (default: 200)",
    )
    args = parser.parse_args()

    if not args.ground_truth.exists():
        print(f"Ground truth not found: {args.ground_truth}")
        print("Create ground truth first:")
        print("  python -m backend.scripts.create_gt")
        print("  OR")
        print("  python -m backend.scripts.prefill_gt")
        sys.exit(1)

    ground_truth = load_ground_truth(args.ground_truth)
    if not ground_truth:
        print("Ground truth is empty. Annotate queries first.")
        sys.exit(1)

    print(f"Loaded {len(ground_truth)} queries from {args.ground_truth}")
    rerank = not args.no_rerank
    mode = "siglip2_rerank" if rerank else "siglip2"
    print(f"Mode: {mode} | Sentence split: always | Fusion: RRF")
    if rerank:
        print(f"Rerank candidates: {args.rerank_candidates}")

    t0 = time.time()
    result = evaluate(
        ground_truth,
        rerank=rerank,
        rerank_candidates=args.rerank_candidates,
    )
    elapsed = round(time.time() - t0, 1)

    print_summary(result["summary"])
    print_per_query_table(result["per_query"])

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "encoder_mode": "siglip2",
            "rerank": rerank,
            "rerank_candidates": args.rerank_candidates if rerank else None,
            "sentence_split": True,
            "fusion": "RRF",
            "mode": mode,
        },
        "summary": result["summary"],
        "per_query": result["per_query"],
        "total_time_s": elapsed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Results saved to {args.output}")
    print(f"Total time: {elapsed}s")


if __name__ == "__main__":
    main()

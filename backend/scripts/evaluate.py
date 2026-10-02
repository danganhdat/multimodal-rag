"""
Evaluation pipeline for AIC25 multimodal search.

Metrics:
  - Mean of Top-k R-Scores (competition metric)
  - Recall@k
  - MRR (Mean Reciprocal Rank)

Usage:
    python -m backend.scripts.evaluate --ground-truth eval/ground_truth.json
    python -m backend.scripts.evaluate --ground-truth eval/ground_truth.json --modes dense,dense_rerank
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

API_BASE = "http://localhost:8000/api"
K_VALUES = [1, 5, 20, 50, 100]


def load_ground_truth(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data:
        if "frame_range" not in item:
            fi = item["frame_idx"]
            item["frame_range"] = [fi - 30, fi + 30]
    return data


def search(query: str, rerank: bool = False, top_k: int = 100) -> list[dict]:
    body = {
        "queries": [{"text": query, "lang": "en"}],
        "config": {
            "top_k": top_k,
            "rerank": rerank,
            "rerank_candidates": 200,
        },
    }
    resp = requests.post(f"{API_BASE}/search", json=body, timeout=60)
    resp.raise_for_status()
    return resp.json()["results"]


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


def evaluate_mode(
    ground_truth: list[dict], rerank: bool, mode_name: str
) -> dict:
    scores = []
    recalls = {k: [] for k in K_VALUES}
    mrr_scores = []

    for i, gt in enumerate(ground_truth):
        query = gt["query"]
        print(f"  [{i + 1}/{len(ground_truth)}] {query[:60]}...", end=" ")

        try:
            results = search(query, rerank=rerank)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        s = mean_topk_rscore(results, gt)
        scores.append(s)

        for k in K_VALUES:
            recalls[k].append(recall_at_k(results, gt, k))

        mrr_scores.append(mrr(results, gt))
        print(f"score={s:.3f}")

    n = len(scores) or 1
    return {
        "mode": mode_name,
        "mean_topk_rscore": sum(scores) / n,
        "mrr": sum(mrr_scores) / n,
        **{f"recall@{k}": sum(recalls[k]) / n for k in K_VALUES},
        "num_queries": len(scores),
    }


def print_table(results: list[dict]) -> None:
    headers = [
        "Mode",
        "TopK-R",
        "MRR",
        *[f"R@{k}" for k in K_VALUES],
    ]
    rows = []
    for r in results:
        rows.append([
            r["mode"],
            f"{r['mean_topk_rscore']:.4f}",
            f"{r['mrr']:.4f}",
            *[f"{r[f'recall@{k}']:.4f}" for k in K_VALUES],
        ])

    col_widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep_line = "-+-".join("-" * w for w in col_widths)

    print()
    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(row, col_widths)))
    print()


def main():
    parser = argparse.ArgumentParser(description="AIC25 Search Evaluation")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("eval/ground_truth.json"),
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="dense,dense_rerank",
        help="Comma-separated modes: dense, dense_rerank",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/results.json"),
    )
    args = parser.parse_args()

    if not args.ground_truth.exists():
        print(f"Ground truth not found: {args.ground_truth}")
        print("Create eval/ground_truth.json with format:")
        print('[{"query": "...", "video_name": "L21_V001", "frame_idx": 123, "frame_range": [100, 150]}]')
        sys.exit(1)

    ground_truth = load_ground_truth(args.ground_truth)
    print(f"Loaded {len(ground_truth)} queries from {args.ground_truth}")

    modes = args.modes.split(",")
    all_results = []

    for mode in modes:
        rerank = mode.endswith("_rerank")
        print(f"\nEvaluating: {mode}")
        t0 = time.time()
        result = evaluate_mode(ground_truth, rerank=rerank, mode_name=mode)
        result["time_seconds"] = round(time.time() - t0, 1)
        all_results.append(result)

    print_table(all_results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

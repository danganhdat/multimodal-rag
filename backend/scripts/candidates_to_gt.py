"""
Convert gt_candidates.json → ground_truth.json by selecting top-1 candidate.

For each query, takes the highest-ranked candidate as the ground truth answer.
Sets frame_range = [frame_idx - 30, frame_idx + 30] as tolerance window.
Uses the original query as eval_query (can be manually rewritten later).

Usage:
    python -m backend.scripts.candidates_to_gt
    python -m backend.scripts.candidates_to_gt --input eval/gt_candidates.json --output eval/ground_truth.json
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Convert candidates to ground truth")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "eval" / "gt_candidates.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "eval" / "ground_truth.json",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=30,
        help="Frame range tolerance (default: ±30 frames)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Candidates file not found: {args.input}")
        print("Run prefill_gt.py first.")
        sys.exit(1)

    candidates = json.loads(args.input.read_text(encoding="utf-8"))
    ground_truth = []
    skipped = 0

    for entry in candidates:
        if not entry.get("candidates"):
            print(f"  SKIP {entry['query_id']}: no candidates")
            skipped += 1
            continue

        top = entry["candidates"][0]
        fi = top["frame_idx"]

        ground_truth.append({
            "query_id": entry["query_id"],
            "query_type": entry["query_type"],
            "query": entry["query"],
            "eval_query": entry.get("eval_query") or entry["query"],
            "video_name": top["video_name"],
            "frame_idx": fi,
            "frame_range": [fi - args.tolerance, fi + args.tolerance],
            "lang": "vi",
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(ground_truth, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Created {len(ground_truth)} ground truth entries ({skipped} skipped)")
    print(f"Saved to {args.output}")
    print()
    print("NOTE: This uses top-1 search results as 'ground truth'.")
    print("For accurate evaluation, manually verify and adjust entries,")
    print("and rewrite eval_query to avoid contamination.")


if __name__ == "__main__":
    main()

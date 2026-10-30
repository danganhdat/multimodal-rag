"""python -m video_rag_eval <stage> [options]."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .core import append_jsonl, config, configured_api_key, output


def main() -> None:
    parser = argparse.ArgumentParser(description="Video RAG silver benchmark pipeline")
    parser.add_argument("stage", choices=["inventory", "verify-index", "reembed", "sample", "facts", "generate",
                                          "verify", "claims", "pool", "freeze", "run", "run-all", "adjudicate", "score", "score-all", "smoke-metrics", "audit", "report"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-videos", type=int, help="Inventory smoke test only; omit for complete corpus")
    parser.add_argument("--max-items", type=int, help="Limit new facts in one invocation for a smoke test")
    parser.add_argument("--source-only", action="store_true", help="Pilot inventory limited to videos with original MP4")
    parser.add_argument("--experiment", choices=[f"E{i}" for i in range(1, 9)])
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Score retrieval only, without API calls")
    args = parser.parse_args()
    cfg = config(args.config)
    try:
        _run(args, cfg)
    except Exception as exc:
        message = str(exc)
        key = configured_api_key()
        if key:
            message = message.replace(key, "[REDACTED]")
        append_jsonl(output(cfg, "derived", "stage_errors.jsonl"),
                     {"stage": args.stage, "time_utc": datetime.now(timezone.utc).isoformat(),
                      "error_type": type(exc).__name__, "message": message})
        raise


def _run(args, cfg):
    if args.stage == "inventory":
        from .corpus import inventory
        result = inventory(cfg, args.max_videos, args.source_only)
    elif args.stage == "verify-index":
        from .corpus import verify_siglip2
        result = verify_siglip2(cfg)
    elif args.stage == "reembed":
        from .corpus import reembed
        result = reembed(cfg)
    elif args.stage == "sample":
        from .corpus import dedup_and_sample
        from .labels import _milvus
        try:
            milvus = _milvus(cfg)
        except Exception:
            milvus = None
        result = dedup_and_sample(cfg, milvus)
    elif args.stage == "facts":
        from .labels import extract_facts
        result = extract_facts(cfg, args.max_items)
    elif args.stage == "generate":
        from .labels import generate_qa
        result = generate_qa(cfg)
    elif args.stage == "verify":
        from .labels import verify_qa
        result = verify_qa(cfg)
    elif args.stage == "pool":
        from .labels import pool_and_label
        result = pool_and_label(cfg)
    elif args.stage == "claims":
        from .labels import verify_claims
        result = verify_claims(cfg)
    elif args.stage == "freeze":
        from .labels import freeze
        result = freeze(cfg, args.allow_partial)
    elif args.stage == "run":
        from .runner import run_experiment
        if not args.experiment:
            parser.error("--experiment E1..E8 required for run")
        result = run_experiment(cfg, args.experiment, args.split)
    elif args.stage == "run-all":
        from .runner import run_all
        result = run_all(cfg, args.split)
    elif args.stage == "adjudicate":
        from .labels import adjudicate_runs
        result = adjudicate_runs(cfg)
    elif args.stage == "score":
        from .scoring import score_runs
        if not args.run:
            parser.error("--run path required for score")
        result = score_runs(cfg, args.run, not args.offline)
    elif args.stage == "smoke-metrics":
        from .scoring import smoke_metrics
        result = smoke_metrics(cfg)
    elif args.stage == "score-all":
        from .scoring import score_all
        result = score_all(cfg)
    elif args.stage == "audit":
        from .documentation import audit
        result = audit(cfg)
    else:
        from .documentation import report
        result = report(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

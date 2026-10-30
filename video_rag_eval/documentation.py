"""Human-readable data report and executable traceability audit."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .core import ROOT, configured_api_key, output, read_jsonl, sha256, write_json
from .ranges import interval
from .scoring import load_ranges

PLAN = ROOT / "docs" / "source-plan-video-rag-eval.md"

EVIDENCE_PATHS = {
    "source_plan": "docs/source-plan-video-rag-eval.md",
    "inventory": "video_rag_eval/corpus.py · derived/frames.parquet · derived/dataset_report.json",
    "frame_ids": "video_rag_eval/corpus.py · tests/test_video_rag_eval.py",
    "index": "video_rag_eval/corpus.py · derived/index_verification.json",
    "api_access": "video_rag_eval/labels.py · derived/api_preflight.json",
    "sample": "video_rag_eval/corpus.py · derived/sample_report.json",
    "facts": "video_rag_eval/labels.py · derived/frame_facts.jsonl",
    "proposals": "video_rag_eval/labels.py · derived/proposals.jsonl",
    "verification": "video_rag_eval/labels.py · derived/judge_votes.jsonl",
    "qrel_pool": "video_rag_eval/labels.py · derived/pool_judgments.jsonl",
    "benchmark_join": "video_rag_eval/labels.py · benchmark/v1/*.jsonl",
    "required_claims": "video_rag_eval/labels.py · benchmark/v1/answers.jsonl · tests/test_video_rag_eval.py",
    "answer_claims": "video_rag_eval/labels.py · derived/claim_verification.jsonl",
    "question_type_quotas": "video_rag_eval/config.yaml · benchmark/v1/queries.jsonl",
    "benchmark_target": "benchmark/v1/benchmark_manifest.json",
    "hashes": "benchmark/v1/benchmark_manifest.json",
    "qrels": "benchmark/v1/qrels.tsv",
    "unjudged": "derived/pool_judgments.jsonl · benchmark/v1/qrels.tsv",
    "evidence_paths": "benchmark/v1/evidence.jsonl · D:/keyframes",
    "ocr_quality": "derived/judge_votes.jsonl · derived/crops",
    "temporal_quality": "derived/judge_votes.jsonl · D:/map-keyframes",
    "pool_diversity": "derived/pool_judgments.jsonl",
    "secrets": ".gitignore · derived/api_usage.jsonl",
    "accepted_ranges": "video_rag_eval/ranges.py · benchmark/v1/accepted_ranges.tsv · tests/test_video_rag_eval.py",
    "split": "video_rag_eval/labels.py · benchmark/v1/splits.json",
    "cross_video_duplicate_split": "video_rag_eval/labels.py · benchmark/v1/splits.json",
    "four_metrics": "video_rag_eval/scoring.py · eval/E*_summary.json",
    "metric_smoke": "video_rag_eval/scoring.py · eval/metric_smoke.json · tests/test_video_rag_eval.py",
    "additional_metrics": "video_rag_eval/scoring.py · eval/E*_scores.jsonl",
    "reference_sources": "docs/video-rag-eval-data-report.md · source references · eval/metric_smoke.json",
    "docs_report": "docs/video-rag-eval-data-report.md",
    **{f"E{i}": f"video_rag_eval/runner.py · runs/E{i}.jsonl" for i in range(1, 9)},
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _status(ok: bool, partial: bool = False) -> str:
    return "đạt" if ok else "một phần" if partial else "chưa đạt"


def audit(cfg: dict) -> dict:
    docs_root = Path(cfg.get("docs_root", ROOT / "docs"))
    docs_root.mkdir(parents=True, exist_ok=True)
    derived = output(cfg, "derived", "dataset_report.json").parent
    benchmark = output(cfg, "benchmark", "v1", "benchmark_manifest.json").parent
    manifest = _read(benchmark / "benchmark_manifest.json")
    frames = read_jsonl(derived / "frames.jsonl")
    facts = read_jsonl(derived / "frame_facts.jsonl")
    proposals = read_jsonl(derived / "proposals.jsonl")
    votes = read_jsonl(derived / "judge_votes.jsonl")
    quarantine = read_jsonl(derived / "quarantine.jsonl")
    pool = read_jsonl(derived / "pool_judgments.jsonl")
    queries = read_jsonl(benchmark / "queries.jsonl")
    answers = read_jsonl(benchmark / "answers.jsonl")
    evidence = read_jsonl(benchmark / "evidence.jsonl")
    qids = {r["query_id"] for r in queries}
    qrel_rows = []
    if (benchmark / "qrels.tsv").exists():
        with (benchmark / "qrels.tsv").open(newline="", encoding="utf-8") as f:
            qrel_rows = list(csv.reader(f, delimiter="\t"))
    ranges = load_ranges(benchmark / "accepted_ranges.tsv") if (benchmark / "accepted_ranges.tsv").exists() else []
    split = _read(benchmark / "splits.json")
    runs = {f"E{i}": read_jsonl(output(cfg, "runs", f"E{i}.jsonl")) for i in range(1, 9)}
    scores = {f"E{i}": _read(output(cfg, "eval", f"E{i}_summary.json")) for i in range(1, 9)}
    checks = {}

    def add(name: str, passed: bool, detail: str, partial: bool = False):
        checks[name] = {"status": _status(passed, partial), "detail": detail}

    add("source_plan", PLAN.is_file(), str(PLAN))
    dataset = _read(derived / "dataset_report.json")
    add("inventory", bool(frames) and dataset.get("videos_scanned") == dataset.get("videos_available"),
        f"{len(frames)} valid frame rows; {dataset.get('videos_scanned', 0)}/{dataset.get('videos_available', 'unknown')} videos scanned; {dataset.get('quarantine_count', 'unknown')} quarantined",
        partial=bool(frames))
    ids = [r["frame_id"] for r in frames]
    add("frame_ids", bool(ids) and len(ids) == len(set(ids)) and all(Path(r["image_path"]).is_file() for r in frames),
        "unique IDs and image paths")
    index = _read(derived / "index_verification.json")
    add("index", index.get("passed") is True, "SigLIP2 parity samples verified" if index else "Milvus/model parity not run")
    api = _read(derived / "api_preflight.json")
    add("api_access", api.get("usable") is True,
        "OpenRouter inference key verified" if api.get("usable") else
        ("configured key is management/provisioning only" if api.get("management_or_provisioning") else "OpenRouter inference access not verified"))
    sample = read_jsonl(derived / "sampled_frames.jsonl")
    sample_report = _read(derived / "sample_report.json")
    add("sample", bool(sample) and sample_report.get("near_dedup_enabled") is True,
        f"{len(sample)} sampled frames; cosine/OCR near-dedup enabled={sample_report.get('near_dedup_enabled', False)}",
        partial=bool(sample))
    add("facts", bool(facts), f"{len(facts)} fact records")
    add("proposals", bool(proposals), f"{len(proposals)} proposals")
    accepted = [r for r in votes if r.get("accepted")]
    add("verification", len(accepted) >= cfg["target_qa"],
        f"{len(accepted)} accepted / {len(votes)} judged, target {cfg['target_qa']}", partial=bool(votes))
    claim_votes = {r["frame_id"]: r for r in read_jsonl(derived / "claim_verification.jsonl")}
    add("answer_claims", bool(evidence) and all(claim_votes.get(e["source_frame_id"], {}).get("passed") is True for e in evidence),
        "frozen QA answer claims verified jointly on ordered source images", partial=bool(claim_votes))
    type_counts = Counter(q["question_type"] for q in queries)
    deficits = {kind: {"actual": type_counts[kind], "target": round(cfg["target_qa"] * fraction)}
                for kind, fraction in cfg["sampling"]["quotas"].items() if type_counts[kind] < round(cfg["target_qa"] * fraction)}
    add("question_type_quotas", bool(queries) and not deficits,
        f"question-type quota deficits: {json.dumps(deficits, ensure_ascii=False)}", partial=bool(queries))
    add("qrel_pool", bool(pool) and any(r.get("status") == "judged" for r in pool),
        f"{sum(r.get('status') == 'judged' for r in pool)} judged / {len(pool)} candidates")
    joins = bool(qids) and len(qids) == len(queries) and qids == {r["query_id"] for r in answers} == {r["query_id"] for r in evidence}
    add("benchmark_join", joins, "query/answer/evidence IDs align")
    add("required_claims", bool(answers) and all(not a.get("answerable") or
        (bool(a.get("required_claims")) and all(not re.fullmatch(r"(?:f|fact)[_ -]?\d+|\d+", str(c).strip(), re.I)
                                              for c in a["required_claims"])) for a in answers),
        "answerable QA has explicit visual claim text rather than unresolved fact IDs")
    add("benchmark_target", len(queries) >= cfg["target_qa"], f"{len(queries)} frozen QA")
    hashes = manifest.get("hashes", {})
    add("hashes", bool(hashes) and all((benchmark / name).is_file() and sha256(benchmark / name) == digest for name, digest in hashes.items()),
        "manifest hashes match frozen files")
    add("qrels", bool(qrel_rows) and all(len(row) == 4 and row[0] in qids and row[1] == "0" and int(row[3]) in range(4) for row in qrel_rows),
        f"{len(qrel_rows)} frame-level TREC qrels")
    source_by_qid = {r["query_id"]: r.get("source_frame_id") or
                     (r["evidence_sets"][0]["frame_ids"][0] if r.get("evidence_sets") else None)
                     for r in evidence}
    judged_pairs = {(r["frame_id"], r["source_frame_id"]) for r in pool if r.get("status") == "judged"}
    add("unjudged", bool(pool) and all((source_by_qid.get(qid), fid) in judged_pairs for qid, _, fid, _ in qrel_rows),
        "unjudged pool candidates excluded from qrels", partial=bool(pool))
    evidence_paths_ok = bool(evidence) and all(
        (not answers or not next((a["answerable"] for a in answers if a["query_id"] == e["query_id"]), False) or e.get("evidence_sets")) and
        all((Path(cfg["sources"]["keyframes"]) / fid.split("/")[0] / (fid.split("/")[1] + ".jpg")).is_file()
            for es in e.get("evidence_sets", []) for fid in es.get("frame_ids", [])) and
        all(Path(p).is_file() for es in e.get("evidence_sets", []) for p in es.get("crop_paths", []))
        for e in evidence)
    add("evidence_paths", evidence_paths_ok, "all referenced images/crops exist and answerable QA has an evidence set")
    ocr_votes = [r for r in votes if r.get("accepted") and r["proposal"].get("question_type") == "ocr_signage"]
    add("ocr_quality", len(ocr_votes) >= 1 and all(r.get("crop_paths") and len(r.get("votes", [])) == 2 for r in ocr_votes),
        f"{len(ocr_votes)} accepted OCR QA with crop and two judges", partial=bool(ocr_votes))
    temporal_votes = [r for r in votes if r.get("accepted") and r["proposal"].get("question_type") == "multi_frame_temporal"]
    add("temporal_quality", len(temporal_votes) >= 1 and all(len(r.get("source_frame_ids", [])) >= 2 for r in temporal_votes),
        f"{len(temporal_votes)} accepted multi-frame QA with timestamped source IDs", partial=bool(temporal_votes))
    pool_sources = {s.split(":")[0] for r in pool for s in r.get("sources", [])}
    add("pool_diversity", {"source", "siglip2", "object", "random"}.issubset(pool_sources),
        f"pool sources: {sorted(pool_sources)}", partial=bool(pool_sources))
    video_map = defaultdict(dict)
    for row in frames:
        video_map[row["video_id"]][row["n"]] = row["frame_idx"]
    valid_ranges = bool(ranges)
    for row in ranges:
        vid, n = row["source_frame_id"].split("/")
        mapping = video_map.get(vid)
        if not mapping:
            # Inventory may be restricted to pilot videos; load the source CSV directly.
            csv_path = Path(cfg["sources"]["maps"]) / f"{vid}.csv"
            if csv_path.exists():
                with csv_path.open(newline="", encoding="utf-8-sig") as f:
                    mapping = {int(r["n"]): int(r["frame_idx"]) for r in csv.DictReader(f)}
        if not mapping or int(n) not in mapping:
            valid_ranges = False
            break
        expected = interval(mapping[int(n)], [mapping[k] for k in sorted(mapping)])
        if expected != (row["lo_frame_idx"], row["hi_frame_idx"]):
            valid_ranges = False
            break
    add("accepted_ranges", valid_ranges, f"{len(ranges)} intervals derived from neighboring frame_idx values")
    dev, test = set(split.get("dev_videos", [])), set(split.get("test_videos", []))
    add("split", bool(split) and not dev.intersection(test) and all(r["source_video_id"] in dev | test for r in evidence),
        f"dev={len(dev)} videos, test={len(test)} videos")
    # Exact cross-video copies must not straddle dev and test.
    digest_splits = defaultdict(set)
    for row in frames:
        if row["video_id"] in dev:
            digest_splits[row["sha256"]].add("dev")
        if row["video_id"] in test:
            digest_splits[row["sha256"]].add("test")
    exact_split_ok = bool(split) and not any(len(s) > 1 for s in digest_splits.values())
    add("cross_video_duplicate_split", False,
        "exact image hashes do not cross dev/test; global cross-video near-scene dedup is not covered by the pilot" if exact_split_ok else
        "split missing or identical image hashes cross dev/test", partial=exact_split_ok)
    run_ids = {r["query_id"] for r in queries if r.get("split") == "test"}
    for exp, rows in runs.items():
        add(exp, bool(run_ids) and {r["query_id"] for r in rows} == run_ids and all(r["query_id"] in run_ids and
                                 r.get("context_frame_ids") == [h["frame_id"] for h in r.get("context_hits", [])]
                                 for r in rows), f"{len(rows)}/{len(run_ids)} test runs; context IDs recorded", partial=bool(rows))
        if exp == "E7" and checks[exp]["status"] == "đạt":
            checks[exp] = {"status": "một phần", "detail": f"{len(rows)}-query run uses per-frame captions plus images; grouped shot-summary-before-agent is not implemented in this pilot"}
    add("four_metrics", bool(run_ids) and all(scores[e].get("queries") == len(run_ids) and all(scores[e].get(k) is not None for k in
                                ("visual_context_recall", "qrels_context_precision_ap",
                                 "ragas_multimodal_faithfulness_binary", "ragas_answer_relevancy")) for e in scores),
        "four primary metrics produced from actual context", partial=any(scores.values()))
    smoke = _read(output(cfg, "eval", "metric_smoke.json"))
    add("metric_smoke", smoke.get("passed") is True,
        "known-right / known-wrong / insufficient-context smoke on real source pixels", partial=bool(smoke))
    extra_keys = {"recall_at_50", "recall_at_300", "ndcg_at_10", "mrr_at_10", "answer_correctness",
                  "citation_accuracy", "negative_rejection", "latency_ms", "estimated_usd_delta"}
    scored_rows = {exp: read_jsonl(output(cfg, "eval", f"{exp}_scores.jsonl")) for exp in runs}
    add("additional_metrics", bool(run_ids) and all(len(rows) == len(run_ids) and all(extra_keys <= set(r) for r in rows)
                                                   for rows in scored_rows.values()),
        "retrieval, correctness, citation, rejection, latency and cost fields are present", partial=any(scored_rows.values()))
    report_path = docs_root / "video-rag-eval-data-report.md"
    add("docs_report", report_path.exists() and "```mermaid" in report_path.read_text(encoding="utf-8"),
        "data-pipeline report with Mermaid diagrams")
    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    add("reference_sources", "https://huggingface.co/google/siglip2-base-patch16-512" in report_text and
        "https://docs.ragas.io/en/v0.4.3/" in report_text and smoke.get("passed") is True,
        "official encoder/RAGAS/OpenRouter references checked and native RAGAS exercised")
    key = configured_api_key()
    generated = [p for p in cfg["output"].rglob("*.json*") if p.is_file() and p.stat().st_size < 10_000_000]
    no_key_leak = not key or all(key not in p.read_text(encoding="utf-8", errors="ignore") for p in generated)
    add("secrets", no_key_leak and "data/" in (ROOT / ".gitignore").read_text(encoding="utf-8"),
        "generated artifacts exclude configured API key; data output is gitignored")
    mapping = {
        "0": ["benchmark_join", "qrels", "index"], "1": ["inventory", "frame_ids"],
        "2": ["benchmark_join", "qrels", "accepted_ranges"],
        "3": ["sample", "api_access", "facts", "proposals", "verification", "qrel_pool", "split"],
        "4": ["metric_smoke", "four_metrics"], "5": ["additional_metrics"],
        "6": ["unjudged", "split", "qrel_pool", "pool_diversity"], "7": ["facts", "proposals", "verification"],
        "8": ["inventory", "api_access", "facts", "proposals", "verification", "qrel_pool", "benchmark_join", "E1", "four_metrics"],
        "9": ["benchmark_target", "four_metrics", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"],
        "10": ["docs_report", "four_metrics"], "11": ["verification", "qrel_pool", "unjudged"],
        "12": ["reference_sources"],
        "M0": ["inventory", "frame_ids"], "M1": ["sample", "question_type_quotas"], "M2": ["api_access", "verification", "answer_claims"],
        "M3": ["qrel_pool", "qrels", "pool_diversity"], "M4": ["benchmark_join", "required_claims", "answer_claims", "hashes", "split", "evidence_paths"],
        "M5": ["metric_smoke", "four_metrics", "additional_metrics"], "M6": ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"],
    }
    for i in range(1, 9):
        mapping[f"E{i}"] = [f"E{i}"]
    gates = [["benchmark_join", "verification"], ["evidence_paths", "verification", "answer_claims"],
             ["pool_diversity", "qrels", "unjudged"], ["qrels", "accepted_ranges"], ["ocr_quality"],
             ["temporal_quality", "index"], ["split", "cross_video_duplicate_split"],
             ["four_metrics"], ["metric_smoke"], ["docs_report"], ["secrets"]]
    for i, dependencies in enumerate(gates, 1):
        mapping[f"QG{i:02d}"] = dependencies
    requirements = []
    for req, dependencies in mapping.items():
        states = [checks[k]["status"] for k in dependencies]
        status = "đạt" if all(s == "đạt" for s in states) else "một phần" if any(s != "chưa đạt" for s in states) else "chưa đạt"
        requirements.append({"plan_section": req, "status": status,
                             "evidence": ", ".join(dependencies),
                             "detail": "; ".join(checks[k]["detail"] for k in dependencies if checks[k]["status"] != "đạt")})
    result = {"source_plan_sha256": sha256(PLAN) if PLAN.is_file() else None,
              "checks": checks, "requirements": requirements}
    write_json(output(cfg, "audit", "requirements.json"), result)
    _write_requirements_doc(result, docs_root)
    return {"passed": sum(c["status"] == "đạt" for c in checks.values()),
            "total": len(checks), "report": str(docs_root / "video-rag-eval-requirements.md")}


def _write_requirements_doc(result: dict, docs_root: Path = ROOT / "docs") -> None:
    lines = ["# Đối chiếu PLAN_VIDEO_RAG_EVAL.md", "",
             f"SHA-256 plan gốc: `{result['source_plan_sha256'] or 'không tìm thấy'}`.", "",
             "Trạng thái được tạo từ artifact/kiểm tra hiện có. `một phần` không được tính là nghiệm thu.", "",
             "| Mục trong plan | Trạng thái | Code / artifact / test | Việc còn thiếu |", "|---|---|---|---|"]
    for row in result["requirements"]:
        detail = row["detail"].replace("|", "\\|").replace("\n", " ")
        locations = "; ".join(dict.fromkeys(EVIDENCE_PATHS.get(key.strip(), key.strip()) for key in row["evidence"].split(",")))
        lines.append(f"| {row['plan_section']} | {row['status']} | {locations} | {detail} |")
    lines += ["", "## Khác biệt có chủ ý", "",
              "- Pilot xác minh vector SigLIP2 trong Milvus trên mẫu; lệnh `reembed` tạo lại toàn bộ embedding khi không khớp.",
              "- Generator và hai judge dùng cùng checkpoint Qwen3.8 Flash với prompt độc lập; đây là giới hạn về tính độc lập của nhãn silver.",
              "- `qrels.tsv` giữ nhãn keyframe; `accepted_ranges.tsv` là phép chiếu sang các frame video được chấp nhận để chấm truy hồi.",
              "- Các metric cần MLLM/RAGAS chỉ được đánh dấu đạt sau khi đã chạy trên ảnh thật.", ""]
    lines += ["- E7 pilot dùng caption từng frame kèm ảnh; đây là proxy của Shot Summary. Bản gom shot và cấp summary trước agent trong plan gốc chưa được triển khai.", "",
              "- Quota loại câu hỏi là mục tiêu lấy mẫu. Các câu không qua gate bị loại; bảng kiểm tra ghi rõ quota thiếu thay vì bù bằng nhãn chưa được kiểm chứng.", ""]
    lines += ["- Split chặn trùng video và ảnh có cùng hash. Dedup gần giống hiện chạy theo video; chưa có kiểm tra near-scene toàn cục giữa các video, nên gate tương ứng giữ trạng thái một phần.", ""]
    (docs_root / "video-rag-eval-requirements.md").write_text("\n".join(lines), encoding="utf-8")


def report(cfg: dict) -> dict:
    from .labels import quality_failures
    derived = output(cfg, "derived", "dataset_report.json").parent
    benchmark = output(cfg, "benchmark", "v1", "benchmark_manifest.json").parent
    dataset, sample = _read(derived / "dataset_report.json"), _read(derived / "sample_report.json")
    votes = read_jsonl(derived / "judge_votes.jsonl")
    quarantine = read_jsonl(derived / "quarantine.jsonl")
    pool = read_jsonl(derived / "pool_judgments.jsonl")
    queries = read_jsonl(benchmark / "queries.jsonl")
    types = dict(Counter(r.get("question_type", "unknown") for r in queries))
    evidence_rows = read_jsonl(benchmark / "evidence.jsonl")
    source_ids = {r["query_id"]: {fid for es in r.get("evidence_sets", []) for fid in es.get("frame_ids", [])}
                  for r in evidence_rows}
    frozen_qrels = []
    if (benchmark / "qrels.tsv").is_file():
        with (benchmark / "qrels.tsv").open(encoding="utf-8", newline="") as f:
            frozen_qrels = [{"query_id": qid, "frame_id": fid, "grade": int(grade)}
                            for qid, _, fid, grade in csv.reader(f, delimiter="\t")]
    grade_counts = dict(Counter(r["grade"] for r in frozen_qrels))
    counts_per_query = Counter(r["query_id"] for r in frozen_qrels)
    outside_positive = sum(r["grade"] >= 2 and r["frame_id"] not in source_ids.get(r["query_id"], set()) for r in frozen_qrels)
    usage = read_jsonl(derived / "api_usage.jsonl")
    stage_cost = defaultdict(float)
    for entry in usage:
        stage_cost[entry.get("stage", "unknown")] += float(entry.get("estimated_usd", 0))
    key_configured = bool(configured_api_key())
    api = _read(derived / "api_preflight.json")
    summaries = {f"E{i}": _read(output(cfg, "eval", f"E{i}_summary.json")) for i in range(1, 9)}
    lines = ["# Báo cáo pipeline tạo dữ liệu Video RAG Eval", "",
             "Benchmark là **silver/pseudo-ground-truth** do MLLM sinh và kiểm chứng. Số liệu bên dưới lấy trực tiếp từ artifact; dấu `—` nghĩa là bước chưa chạy.", "",
             "Nguồn yêu cầu: [bản sao plan gốc](source-plan-video-rag-eval.md). Dữ liệu gốc đọc tại `D:/keyframes`, `D:/map-keyframes`, `D:/video`; output tại `data/video_rag_eval`.", "",
             "## Luồng tạo benchmark", "", "```mermaid", "flowchart LR",
             "  A[Video + Keyframes + CSV + Objects] --> B[P0 Inventory và xác minh ID/timestamp]",
             "  B --> C[P1 Dedup có bảo vệ OCR và lấy mẫu]",
             "  C --> D[P2 Trích visual facts và crop]",
             "  D --> E[P3 Sinh description, question, answer]",
             "  E --> F{P4 Hai judge độc lập về lượt gọi}",
             "  F -->|accept| C2[Tách claim từ đáp án và kiểm chứng trên chuỗi ảnh nguồn]",
             "  C2 --> G[P5 Pool SigLIP2 + object + random và gán qrels]",
             "  F -->|reject| Q[Quarantine và lý do]",
             "  G --> H[P6 Split theo video và đóng băng]",
             "  H --> I[Queries + Qrels + Answers + Evidence + Accepted ranges]",
             "  I --> J[E1-E8 runs và context thực nhận]",
             "  J --> K[Retrieval metrics + visual judges + RAGAS]", "```", "",
             "## Dòng dữ liệu và khoảng chấp nhận", "", "```mermaid", "flowchart LR",
             "  M[CSV: n, pts_time, fps, frame_idx] --> F[frame_id: video/n]",
             "  F --> Q[qrels.tsv: query_id, frame_id, grade]",
             "  M --> R[accepted_ranges.tsv: video_id, lo, hi]",
             "  Q --> R", "  F --> E[evidence.jsonl: evidence_sets]",
             "  R --> S[Recall, nDCG, MRR, answer matching]",
             "  E --> V[Visual Context Recall và Faithfulness trên ảnh thật]", "```", "",
             "Với keyframe `frame_idx=351`, hai keyframe liền kề có `frame_idx=261` và `411`, nên khoảng đóng là `[262,410]`. Hit phải cùng `video_id`. Khoảng chồng nhau được gộp khi chấm để tránh cộng điểm lặp; qrels gốc vẫn giữ nguyên để truy vết. Keyframe đầu/cuối dùng giới hạn có trong CSV khi không có số frame cuối video đáng tin.", "",
             "## Cách gán qrels và bằng chứng temporal", "",
             f"Mỗi QA lấy hợp frame nguồn, top {cfg['pool']['dense_top']} SigLIP2, tối đa {cfg['pool']['auxiliary_top']} ứng viên cùng object và {cfg['pool']['random_negatives']} frame ngẫu nhiên. ID trùng được loại trước khi chấm. Nhãn được ghi ngay vào `derived/pool_judgments.jsonl`; pool hữu hạn không chứng minh exhaustive relevance toàn corpus.", "",
             "Mỗi lượt qrel xem **toàn bộ câu hỏi + các claim + một candidate frame**, có thể thêm crop của chính frame nguồn. Chưa tách câu hỏi thành sub-query; lượt qrel không xem cả chuỗi. Grade 3 trực tiếp trả lời; grade 2 gần đủ hoặc hỗ trợ một sự kiện của câu temporal; grade 1 cùng chủ đề; grade 0 đã judge là không liên quan. Grade 3 có lượt xác nhận thứ hai và đối chiếu đáp án theo đúng câu hỏi. `uncertain` hoặc bất đồng giữ là unjudged.", "",
             "Hai frame grade 2 không tự chứng minh quan hệ thời gian. Chuỗi nguồn được judge chung ở bước xác minh QA/claim. Khi đánh giá, Context Recall và Faithfulness xem các ảnh/crop thực sự được cấp, kiểm tra cả hai trạng thái và quan hệ thứ tự. `supports_claim_ids` chỉ tham chiếu các claim mà lượt qrel đã nhận; `judged_claims` giữ nội dung claim của lượt đó khi claim được chuẩn hóa sau pilot.", "",
             "## Chạy lại pipeline", "", "```powershell",
             ".venv\\Scripts\\python.exe -m video_rag_eval inventory --source-only",
             ".venv\\Scripts\\python.exe -m video_rag_eval verify-index",
             ".venv\\Scripts\\python.exe -m video_rag_eval sample",
             ".venv\\Scripts\\python.exe -m video_rag_eval facts",
             ".venv\\Scripts\\python.exe -m video_rag_eval generate",
             ".venv\\Scripts\\python.exe -m video_rag_eval verify",
             ".venv\\Scripts\\python.exe -m video_rag_eval claims",
             ".venv\\Scripts\\python.exe -m video_rag_eval pool",
             ".venv\\Scripts\\python.exe -m video_rag_eval freeze",
             ".venv\\Scripts\\python.exe -m video_rag_eval smoke-metrics",
             "1..8 | ForEach-Object { .venv\\Scripts\\python.exe -m video_rag_eval run --experiment \"E$_\" }",
             ".venv\\Scripts\\python.exe -m video_rag_eval adjudicate",
             "1..8 | ForEach-Object { .venv\\Scripts\\python.exe -m video_rag_eval score --run \"data/video_rag_eval/runs/E$_.jsonl\" }",
             ".venv\\Scripts\\python.exe -m video_rag_eval report",
             ".venv\\Scripts\\python.exe -m video_rag_eval audit", "```", "",
             "Các bước gọi MLLM cần `LLM_API_KEY` trong môi trường. CLI dừng khi thiếu key; không ghi key vào artifact. `--source-only` là phạm vi pilot; bỏ cờ để kiểm kê toàn corpus.", "",
             "## Kiểm kê và chất lượng nhãn", "",
             "| Chỉ số | Giá trị |", "|---|---:|",
             f"| Video đã quét | {dataset.get('videos_scanned', '—')} |",
             f"| Frame hợp lệ | {dataset.get('frames_valid', '—')} |",
             f"| Frame/map quarantine | {dataset.get('quarantine_count', '—')} |",
             f"| Video có file gốc | {dataset.get('source_video_count', '—')} |",
             f"| Nhóm dedup | {sample.get('groups', '—')} |",
             f"| Frame lấy mẫu | {sample.get('sampled', '—')} |",
             f"| Video có mẫu | {sample.get('sampled_videos', '—')} |",
             f"| Mẫu có OCR | {sample.get('sampled_ocr', '—')} |",
             f"| Mẫu có object | {sample.get('sampled_with_objects', '—')} |",
             f"| Proposal được judge | {len(votes) if votes else '—'} |",
             f"| QA chấp nhận | {sum(r.get('accepted', False) for r in votes) if votes else '—'} |",
             f"| QA từ chối | {sum(not r.get('accepted', False) for r in votes) if votes else '—'} |",
             f"| Candidate qrels đã chấm | {sum(r.get('status') == 'judged' for r in pool) if pool else '—'} |",
             f"| Candidate unjudged | {sum(r.get('status') != 'judged' for r in pool) if pool else '—'} |",
             f"| Benchmark đã đóng băng | {len(queries) if queries else '—'} |", "",
             f"Phân bố loại câu hỏi: `{json.dumps(types, ensure_ascii=False) if types else 'chưa có'}`.", "",
             f"Phân bố qrels đã đóng băng theo grade: `{json.dumps(grade_counts) if frozen_qrels else 'chưa có'}`. Số nhãn/query: **{min(counts_per_query.values()) if counts_per_query else '—'}–{max(counts_per_query.values()) if counts_per_query else '—'}**; positive ngoài frame nguồn/evidence gốc: **{outside_positive if frozen_qrels else '—'}**.", "",
             f"Judged coverage của pool đã lấy: **{sum(r.get('status') == 'judged' for r in pool) / len(pool):.1%}** (chỉ trên candidate pool, không phải độ phủ toàn corpus)." if pool else "Judged coverage: chưa có pool.", "",
             f"Quality gates từ chối QA: `{json.dumps(dict(Counter(reason for r in votes if not r.get('accepted') for reason in quality_failures(r))), ensure_ascii=False) if votes else 'chưa có'}`.", "",
             f"Số QA có bất đồng giữa hai judge về ít nhất một cờ chất lượng: **{sum(len(r.get('votes', [])) == 2 and any(r['votes'][0].get(k) != r['votes'][1].get(k) for k in ('answerable', 'description_grounded', 'question_unambiguous', 'leaks_answer', 'needs_more_pixels')) for r in votes)}**.", "",
             f"Quarantine theo nguyên nhân: `{json.dumps(dict(Counter(r.get('reason', 'unknown') for r in quarantine)), ensure_ascii=False) if quarantine else 'không có'}`.", "",
             f"Chi phí theo stage (ước tính USD): `{json.dumps({k: round(v, 5) for k, v in stage_cost.items()}, ensure_ascii=False) if stage_cost else 'chưa có'}`.", "",
             f"Model: `{cfg['model']}`; hai lượt judge dùng cùng checkpoint, prompt độc lập. OpenRouter key: **{'đã cấu hình' if key_configured else 'chưa cấu hình'}**; inference access: **{'đã xác minh' if api.get('usable') else 'chưa xác minh' if not api else 'không dùng được'}**. Chi phí ước tính: **${sum(r.get('estimated_usd', 0) for r in usage):.4f} USD**; giá API có thể đổi.", "",
             "Lý do không chạy pilot: key hiện tại là management/provisioning, endpoint chat trả HTTP 401. Cần thay bằng inference API key trong `.env` hoặc môi trường; không ghi giá trị key vào artifact." if api.get("management_or_provisioning") else "", "",
             "## Kết quả E1–E8", "",
             "| Cấu hình | Context Recall | Context Precision | Faithfulness | Answer Relevance | #queries |",
             "|---|---:|---:|---:|---:|---:|"]
    for exp, row in summaries.items():
        cell = lambda key: f"{row[key]:.3f}" if isinstance(row.get(key), (float, int)) else "—"
        lines.append(f"| {exp} | {cell('visual_context_recall')} | {cell('qrels_context_precision_ap')} | "
                     f"{cell('ragas_multimodal_faithfulness_binary')} | {cell('ragas_answer_relevancy')} | {row.get('queries', '—')} |")
    lines += ["", "Số query có điểm hợp lệ được báo riêng; mẫu unanswerable không có Context Recall tham chiếu, và context còn unjudged không nhận AP giả bằng 0.", "",
              "| Cấu hình | N Recall | N Precision | N Faithfulness | N Relevance |", "|---|---:|---:|---:|---:|"]
    for exp, row in summaries.items():
        counts = row.get("metric_valid_queries", {})
        lines.append(f"| {exp} | {counts.get('visual_context_recall', '—')} | {counts.get('qrels_context_precision_ap', '—')} | "
                     f"{counts.get('ragas_multimodal_faithfulness_binary', '—')} | {counts.get('ragas_answer_relevancy', '—')} |")
    lines += ["", "## Metric bổ sung", "",
              "| Cấu hình | Recall@50 | nDCG@10 | Answer correctness | Citation accuracy | Rejection | P95 ms | Cost USD |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for exp, row in summaries.items():
        cell = lambda key: f"{row[key]:.3f}" if isinstance(row.get(key), (float, int)) else "—"
        lines.append(f"| {exp} | {cell('recall_at_50')} | {cell('ndcg_at_10')} | {cell('answer_correctness')} | "
                     f"{cell('citation_accuracy')} | {cell('negative_rejection')} | {cell('latency_p95_ms')} | "
                     f"{cell('estimated_usd_total')} |")
    lines += ["", "Chi phí trong bảng gồm các lượt model của run, kể cả controller/VLM của E8; các lượt judge/RAGAS nằm trong chi phí theo stage phía trên. Query variants được cache dùng chung, nên chi phí tạo cache thuộc cấu hình chạy đầu tiên.", "",
              "## Theo loại câu hỏi", "",
              "| Cấu hình | Loại | #queries | Context Recall | Context Precision | Correctness |",
              "|---|---|---:|---:|---:|---:|"]
    for exp, row in summaries.items():
        for kind, group in row.get("by_question_type", {}).items():
            cell = lambda key: f"{group[key]:.3f}" if isinstance(group.get(key), (float, int)) else "—"
            lines.append(f"| {exp} | {kind} | {group['queries']} | {cell('visual_context_recall')} | "
                         f"{cell('qrels_context_precision_ap')} | {cell('answer_correctness')} |")
    lines += ["", "Context Recall là visual claim recall; Context Precision là AP từ qrels với ngưỡng relevance ≥2. Faithfulness là RAGAS multimodal binary; Answer Relevance là RAGAS AnswerRelevancy với SigLIP2 text embedding cục bộ. nDCG/MRR dùng thứ hạng đã bỏ kết quả unjudged, đồng thời báo judged coverage.", "",
              "## Giới hạn", "",
              "- Khoảng frame là dung sai chấm truy hồi, không xác nhận mọi ảnh trong khoảng chứa cùng bằng chứng.",
              "- Pool hữu hạn không exhaustively phán nhãn toàn corpus; unjudged không phải negative.",
              "- Cùng checkpoint Qwen cho các judge có thể tạo sai lệch chung.",
              "- AnswerRelevancy dùng SigLIP2 text embedding cùng họ với retriever; diễn giải kết quả với giới hạn phụ thuộc encoder.",
              "- Xem [bảng đối chiếu yêu cầu](video-rag-eval-requirements.md) để biết các bước chưa đạt.", ""]
    lines += ["## Nguồn tham chiếu đã kiểm tra", "",
              "- [Google SigLIP2 model card và cách dùng AutoModel/AutoProcessor](https://huggingface.co/google/siglip2-base-patch16-512). Hash model/processor cục bộ và parity Milvus nằm trong `derived/index_verification.json`.",
              "- [RAGAS 0.4.3 MultiModalFaithfulness](https://docs.ragas.io/en/v0.4.3/concepts/metrics/available_metrics/multi_modal_faithfulness/). Smoke artifact xác nhận verdict binary trên ảnh thật.",
              "- [OpenRouter Qwen3.8 Flash](https://openrouter.ai/qwen/qwen3.8-flash) và [reasoning budget](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens). Ledger dùng giá input/output 0,15/0,47 USD mỗi triệu token; chi phí RAGAS là allowance ước tính.", ""]
    dest = ROOT / "docs" / "video-rag-eval-data-report.md"
    dest.write_text("\n".join(lines), encoding="utf-8")
    return {"report": str(dest), "queries": len(queries), "summaries": sum(bool(s) for s in summaries.values())}

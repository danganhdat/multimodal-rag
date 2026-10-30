"""Range-aware retrieval metrics and actual-context answer evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path

from .core import append_jsonl, frozen_benchmark, output, read_jsonl, sha256, source, write_json, write_jsonl
from .ranges import _disjoint, match

_embedding_encoder = None
_embedding_lock = threading.RLock()


def load_ranges(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{**r, "lo_frame_idx": int(r["lo_frame_idx"]),
                 "hi_frame_idx": int(r["hi_frame_idx"]), "grade": int(r["grade"])}
                for r in csv.DictReader(f, delimiter="\t")]


def _hit_id(hit: dict) -> tuple[str, int]:
    if "video_id" in hit and "frame_idx" in hit:
        return hit["video_id"], int(hit["frame_idx"])
    if "video_name" in hit and "frame_idx" in hit:
        return hit["video_name"], int(hit["frame_idx"])
    raise ValueError("retrieval hit needs video_id/video_name and frame_idx")


def retrieval_scores(qid: str, hits: list[dict], ranges: list[dict], threshold: int = 2,
                     ks: tuple[int, ...] = (10, 50, 300)) -> dict:
    """Condensed-list metrics: unjudged hits are reported, never assigned grade zero."""
    segments = [r for r in _disjoint(ranges) if r["query_id"] == qid]
    relevant = [r for r in segments if r["grade"] >= threshold]
    judged = []
    covered = set()
    for hit in hits:
        vid, frame = _hit_id(hit)
        grade = match(qid, vid, frame, ranges)
        if grade is None:
            continue
        judged.append((vid, frame, grade))
    result = {"judged_coverage": len(judged) / len(hits) if hits else 0.0,
              "unjudged_count": len(hits) - len(judged), "positive_range_units": len(relevant)}
    for k in ks:
        top = hits[:k]
        result[f"judged_coverage_at_{k}"] = sum(match(qid, *_hit_id(h), ranges) is not None for h in top) / len(top) if top else 0.0
        used = set()
        gains = []
        positives = 0
        first_grade3 = None
        judged_rank = 0
        for hit in top:
            vid, frame = _hit_id(hit)
            grade = match(qid, vid, frame, ranges)
            if grade is None:
                continue
            judged_rank += 1
            if grade == 3 and first_grade3 is None:
                first_grade3 = judged_rank
            segment_key = next((i for i, r in enumerate(segments) if
                                r["video_id"] == vid and r["lo_frame_idx"] <= frame <= r["hi_frame_idx"]), None)
            fresh = segment_key is not None and segment_key not in used
            if fresh:
                used.add(segment_key)
                if grade >= threshold:
                    positives += 1
            gain = (2 ** grade - 1) if fresh else 0
            gains.append((judged_rank, gain))
        result[f"recall_at_{k}"] = positives / len(relevant) if relevant else None
        if k == 10:
            dcg = sum(gain / math.log2(rank + 1) for rank, gain in gains)
            ideal = sorted((2 ** r["grade"] - 1 for r in segments), reverse=True)[:k]
            idcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(ideal))
            result["ndcg_at_10"] = dcg / idcg if idcg else None
            result["mrr_at_10"] = 1 / first_grade3 if first_grade3 else 0.0
    return result


def context_precision(qid: str, contexts: list[dict], ranges: list[dict], threshold: int = 2) -> dict:
    grades = [match(qid, *_hit_id(hit), ranges) for hit in contexts]
    coverage = sum(g is not None for g in grades) / len(grades) if grades else 0.0
    if any(g is None for g in grades):
        return {"qrels_context_precision_ap": None, "precision_at_k": None,
                "context_judged_coverage": coverage}
    segments = [r for r in _disjoint(ranges) if r["query_id"] == qid]
    credited = set()
    relevant = []
    for hit, grade in zip(contexts, grades):
        vid, frame = _hit_id(hit)
        key = next((i for i, r in enumerate(segments) if r["video_id"] == vid and
                    r["lo_frame_idx"] <= frame <= r["hi_frame_idx"]), None)
        fresh = key is not None and key not in credited
        if fresh:
            credited.add(key)
        relevant.append(int(grade >= threshold and fresh))
    found = 0
    ap = 0.0
    for rank, positive in enumerate(relevant, 1):
        found += positive
        if positive:
            ap += found / rank
    return {"qrels_context_precision_ap": ap / max(1, found),
            "precision_at_k": found / len(contexts) if contexts else 0.0,
            "context_judged_coverage": coverage}


def answer_correct(answer: str, reference: str) -> bool:
    normalize = lambda s: re.sub(r"\W+", "", s.casefold(), flags=re.UNICODE)
    return bool(normalize(reference)) and normalize(answer) == normalize(reference)


def score_runs(cfg: dict, run_path: Path, with_mllm: bool = True) -> dict:
    from .labels import VisionClient, _answers_equivalent
    from .models import FaithfulnessResponse, SupportResponse
    from .corpus import load_maps
    benchmark = frozen_benchmark(cfg)
    queries = {r["query_id"]: r for r in read_jsonl(benchmark / "queries.jsonl")}
    answers = {r["query_id"]: r for r in read_jsonl(benchmark / "answers.jsonl")}
    evidence = {r["query_id"]: r for r in read_jsonl(benchmark / "evidence.jsonl")}
    ranges = load_ranges(benchmark / "accepted_ranges.tsv")
    supplemental = output(cfg, "eval", "supplemental_ranges.tsv")
    if supplemental.exists():
        ranges.extend(load_ranges(supplemental))
    frames = {r["frame_id"]: r for r in read_jsonl(output(cfg, "derived", "frames.jsonl"))}
    maps, _ = load_maps(source(cfg, "maps"))
    runs = read_jsonl(run_path)
    client = VisionClient(cfg) if with_mllm else None
    out = output(cfg, "eval", run_path.stem + "_scores.jsonl")
    mode = "multimodal" if with_mllm else "offline"
    protocol = hashlib.sha256(json.dumps({"ranges": ranges, "evaluation": cfg["evaluation"],
                                         "model": cfg["model"], "implementation": sha256(Path(__file__))},
                                        sort_keys=True).encode()).hexdigest()
    cached = {(r.get("query_id"), r.get("run_signature"), r.get("scoring_mode"), r.get("protocol_hash")): r
              for r in read_jsonl(out)}
    scores = []
    for run in runs:
        qid = run["query_id"]
        signature = hashlib.sha256(json.dumps(run, sort_keys=True).encode()).hexdigest()
        cache_key = (qid, signature, mode, protocol)
        if cache_key in cached:
            scores.append(cached[cache_key])
            continue
        if qid not in queries:
            raise ValueError(f"unknown query_id {qid}")
        query, answer = queries[qid], answers[qid]
        initial = run.get("retrieved_hits", [])
        contexts = run.get("context_hits", [])
        def cited_grade(fid: str) -> int | None:
            try:
                video, n = fid.split("/")
                return match(qid, video, maps[video][int(n)]["frame_idx"], ranges)
            except (ValueError, KeyError):
                return None
        citations = run.get("citations", [])
        citation_grades = [cited_grade(fid) for fid in citations]
        text_right = answer_correct(run.get("generated_answer", ""), answer["reference_answer"])
        answer_comparison = None
        if client and answer["answerable"] and run.get("generated_answer", "").strip() and not text_right:
            answer_comparison = _answers_equivalent(client, answer["reference_answer"],
                                                     [run["generated_answer"]], "answer_correctness_equivalence", query["question_vi"])
            text_right = answer_comparison["equivalent"] is True
        evidence_sets = evidence[qid].get("evidence_sets", [])
        def covers_source(source_id: str) -> bool:
            source_ranges = [r for r in ranges if r["query_id"] == qid and r["source_frame_id"] == source_id]
            for fid in citations:
                try:
                    video, n = fid.split("/")
                    frame = maps[video][int(n)]["frame_idx"]
                except (ValueError, KeyError):
                    continue
                if any(r["video_id"] == video and r["lo_frame_idx"] <= frame <= r["hi_frame_idx"] for r in source_ranges):
                    return True
            return False
        grounded_citation = any(all(covers_source(fid) for fid in es["frame_ids"]) for es in evidence_sets)
        temporal = any(len(es["frame_ids"]) > 1 for es in evidence_sets)
        ordered_citations = []
        for fid in citations:
            try:
                video, n = fid.split("/")
                ordered_citations.append((video, maps[video][int(n)]["frame_idx"]))
            except (ValueError, KeyError):
                pass
        temporal_order = (all(a[0] == b[0] and a[1] < b[1] for a, b in zip(ordered_citations, ordered_citations[1:]))
                          and len(ordered_citations) >= 2) if temporal else None
        row = {"query_id": qid, "config_id": run["config_id"],
               **retrieval_scores(qid, initial, ranges),
               **context_precision(qid, contexts, ranges, cfg["evaluation"]["relevant_threshold"]),
               "answer_text_correctness": int(text_right) if answer["answerable"] else None,
               "answer_correctness": int(text_right and grounded_citation) if answer["answerable"] else None,
               "answer_semantic_comparison": answer_comparison,
               "temporal_chain_accuracy": int(grounded_citation and temporal_order) if temporal else None,
               "negative_rejection": int(run.get("status") in ("uncertain", "not_found", "clarify")) if not answer["answerable"] else None,
               "latency_ms": run.get("latency_ms"),
               "estimated_usd_delta": run.get("estimated_usd_delta"),
               "distinct_video_diversity_at_10": len({h["video_id"] for h in initial[:10]}) / len(initial[:10]) if initial else 0.0}
        relevant_grade = 2 if temporal else 3
        row["citation_accuracy"] = sum(grade is not None and grade >= relevant_grade for grade in citation_grades) / len(citation_grades) if citation_grades else None
        image_paths = [Path(p) for p in run.get("context_image_paths", []) if Path(p).is_file()]
        if not image_paths:
            for h in contexts:
                fid = h.get("frame_id", "")
                if fid in frames:
                    image = Path(frames[fid]["image_path"])
                elif "/" in fid:
                    video, n = fid.split("/", 1)
                    image = source(cfg, "keyframes") / video / f"{n}.jpg"
                else:
                    continue
                if image.is_file():
                    image_paths.append(image)
        if client and image_paths:
            claims = answer.get("required_claims", [])
            if claims and answer["answerable"]:
                response = client.ask("visual_context_recall", "For each reference claim, judge whether the ordered supplied images directly support it, "
                                      "including any temporal relation requiring multiple frames. Do not use world knowledge. "
                                      f"Question: {query['question_vi']}. Frame IDs: {run.get('context_frame_ids', [])}. "
                                      f"Claims: {json.dumps(claims, ensure_ascii=False)}. Return {{supported:[boolean]}}.", image_paths, SupportResponse)
                support = response.get("supported", [])
                if len(support) != len(claims):
                    raise ValueError("visual context judge returned incorrect number of claims")
                row["visual_context_recall"] = sum(x is True for x in support[:len(claims)]) / len(claims)
                row["visual_context_claim_votes"] = [{"claim": claim, "supported": verdict}
                                                     for claim, verdict in zip(claims, support)]
            else:
                row["visual_context_recall"] = None
            if run.get("generated_answer"):
                response = client.ask("visual_faithfulness", "Break answer into factual claims and judge each strictly from supplied images. "
                                      f"Also consider the actual supplied context texts: {json.dumps(run.get('context_texts', []), ensure_ascii=False)}. "
                                      f"Answer: {run['generated_answer']}. Return {{claims:[string],claim_support:[boolean]}} "
                                      "with one verdict per claim in the same order.", image_paths, FaithfulnessResponse)
                support = response.get("claim_support", [])
                row["visual_claim_faithfulness_ratio"] = sum(x is True for x in support) / len(support) if support else None
                row["visual_faithfulness_claim_votes"] = [{"claim": claim, "supported": verdict}
                                                         for claim, verdict in zip(response["claims"], support)]
            row.update(_ragas_scores(cfg, query, run, image_paths))
        else:
            row.update({"visual_context_recall": 0.0 if with_mllm and answer["answerable"] else None,
                        "visual_claim_faithfulness_ratio": 0.0 if with_mllm and run.get("generated_answer") else None,
                        "ragas_multimodal_faithfulness_binary": None, "ragas_answer_relevancy": None})
        row.update({"run_signature": signature, "scoring_mode": mode, "protocol_hash": protocol})
        scores.append(row)
        append_jsonl(out, row)
    write_jsonl(out, scores)
    summary = {}
    valid_counts = {}
    numeric = sorted({k for r in scores for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    for key in numeric:
        values = [r[key] for r in scores if isinstance(r.get(key), (int, float))]
        summary[key] = sum(values) / len(values) if values else None
        valid_counts[key] = len(values)
    summary.update({"config_id": run_path.stem, "queries": len(scores), "with_mllm": with_mllm})
    for key in ("visual_context_recall", "qrels_context_precision_ap", "ragas_multimodal_faithfulness_binary", "ragas_answer_relevancy"):
        valid_counts.setdefault(key, 0)
    summary["metric_valid_queries"] = valid_counts
    summary["estimated_usd_total"] = sum(float(r.get("estimated_usd_delta") or 0) for r in scores)
    by_type = {}
    for question_type in sorted({queries[r["query_id"]].get("question_type", "unknown") for r in scores}):
        group = [r for r in scores if queries[r["query_id"]].get("question_type", "unknown") == question_type]
        metrics = {}
        for key in ("visual_context_recall", "qrels_context_precision_ap",
                    "ragas_multimodal_faithfulness_binary", "ragas_answer_relevancy",
                    "recall_at_50", "ndcg_at_10", "answer_correctness", "negative_rejection"):
            values = [r[key] for r in group if isinstance(r.get(key), (int, float))]
            metrics[key] = sum(values) / len(values) if values else None
        by_type[question_type] = {"queries": len(group), **metrics}
    summary["by_question_type"] = by_type
    latencies = sorted(r["latency_ms"] for r in scores if isinstance(r.get("latency_ms"), (int, float)))
    if latencies:
        summary["latency_p50_ms"] = latencies[(len(latencies) - 1) // 2]
        summary["latency_p95_ms"] = latencies[math.ceil(0.95 * len(latencies)) - 1]
    write_json(output(cfg, "eval", run_path.stem + "_summary.json"), summary)
    return summary


def _ragas_scores(cfg: dict, query: dict, run: dict, images: list[Path]) -> dict:
    """Use the native Collections metric on actual image paths; fail visibly if unavailable."""
    if not run.get("generated_answer", "").strip():
        return {"ragas_multimodal_faithfulness_binary": None, "ragas_answer_relevancy": None}
    import asyncio
    from openai import AsyncOpenAI
    from ragas.llms.base import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, MultiModalFaithfulness
    from ragas.embeddings.base import BaseRagasEmbedding
    from backend.app.encoder import SigLIP2Encoder
    from .core import CostLedger, api_key, ssl_verify
    import httpx

    class MultilingualEmbeddings(BaseRagasEmbedding):
        def __init__(self):
            global _embedding_encoder
            super().__init__()
            with _embedding_lock:
                if _embedding_encoder is None:
                    _embedding_encoder = SigLIP2Encoder(cfg["milvus"]["siglip2_model"])
                    _embedding_encoder._load()
                self.model = _embedding_encoder

        def embed_text(self, text: str, **kwargs) -> list[float]:
            with _embedding_lock:
                return self.model.encode_text(text)

        async def aembed_text(self, text: str, **kwargs) -> list[float]:
            return await asyncio.to_thread(self.embed_text, text, **kwargs)

    async def evaluate():
        api = AsyncOpenAI(base_url=cfg["base_url"], api_key=api_key(),
                          http_client=httpx.AsyncClient(verify=ssl_verify()), timeout=90, max_retries=4)
        llm = llm_factory(cfg["model"], client=api, max_tokens=3000, temperature=0,
                          extra_body={"reasoning": {"max_tokens": cfg.get("reasoning_max_tokens", 512)}})
        try:
            faith = MultiModalFaithfulness(llm=llm)
            result = await faith.ascore(response=run.get("generated_answer", ""),
                                        retrieved_contexts=[str(p.resolve()) for p in images] + run.get("context_texts", []))
            if run.get("generated_answer", "").strip():
                relevance = AnswerRelevancy(llm=llm, embeddings=MultilingualEmbeddings())
                rel_result = await relevance.ascore(user_input=query["question_vi"],
                                                    response=run["generated_answer"])
                rel = float(rel_result.value)
            else:
                rel = None
            return {"ragas_multimodal_faithfulness_binary": float(result.value),
                    "ragas_multimodal_faithfulness_reason": result.reason,
                    "ragas_answer_relevancy": rel}
        finally:
            await api.close()
    ledger = CostLedger(cfg)
    ledger.reserve_estimate("ragas_faithfulness_and_answer_relevancy", 0.05, cfg["model"])
    return asyncio.run(evaluate())


def score_all(cfg: dict) -> dict:
    paths = [output(cfg, "runs", f"E{i}.jsonl") for i in range(1, 9)]
    if any(not p.is_file() for p in paths):
        raise RuntimeError("all E1-E8 run files are required before score-all")
    results, failure = [], None
    with ThreadPoolExecutor(max_workers=min(2, cfg.get("api_workers", 2))) as executor:
        futures = [executor.submit(score_runs, cfg, path) for path in paths]
        for future in as_completed(futures):
            if future.cancelled():
                continue
            try:
                summary = future.result()
                result = {"config_id": summary["config_id"], "queries": summary["queries"]}
                results.append(result)
                print(json.dumps(result), flush=True)
            except Exception as error:
                failure = failure or error
                for other in futures:
                    other.cancel()
        if failure:
            raise failure
    return {"experiments": sorted(results, key=lambda r: r["config_id"])}


def smoke_metrics(cfg: dict) -> dict:
    """Exercise native RAGAS on real source pixels before scoring all experiments."""
    from .labels import VisionClient
    from .models import SupportResponse, FaithfulnessResponse

    benchmark = frozen_benchmark(cfg)
    queries = read_jsonl(benchmark / "queries.jsonl")
    answers = {r["query_id"]: r for r in read_jsonl(benchmark / "answers.jsonl")}
    evidence = {r["query_id"]: r for r in read_jsonl(benchmark / "evidence.jsonl")}
    query = next(q for q in queries if q["answerable"] and q["question_type"] == "visible_action")
    qid = query["query_id"]
    ref = answers[qid]
    fid = evidence[qid]["source_frame_id"]
    video, n = fid.split("/")
    paths = [source(cfg, "keyframes") / video / f"{n}.jpg"]
    paths.extend(Path(p) for es in evidence[qid]["evidence_sets"] for p in es.get("crop_paths", []))
    path = output(cfg, "eval", "metric_smoke.json")
    digest = sha256(benchmark / "benchmark_manifest.json")
    result = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if result.get("benchmark_manifest_sha256") != digest:
        result = {"benchmark_manifest_sha256": digest, "query_id": qid, "frame_id": fid, "cases": {}}
    client = VisionClient(cfg)
    cases = {"known_right": (ref["reference_answer"], paths),
             "known_wrong": ("Một con rồng màu tím đang bay và phun lửa ngay giữa khung hình.", paths),
             "insufficient_context": (ref["reference_answer"], [])}
    for name, (answer, images) in cases.items():
        if name in result["cases"]:
            continue
        row = {"generated_answer": answer, "context_image_paths": [str(p) for p in images]}
        if images:
            support = client.ask("smoke_context_recall", "Judge each reference claim strictly from the supplied images. "
                                 f"Claims: {json.dumps(ref['required_claims'], ensure_ascii=False)}. "
                                 "Return {supported:[boolean]}.", images, SupportResponse)["supported"]
            if len(support) != len(ref["required_claims"]):
                raise ValueError("smoke context recall returned the wrong claim count")
            row["visual_context_recall"] = sum(support) / len(support)
            faith = client.ask("smoke_visual_faithfulness", "Split the answer into factual claims and judge support "
                               f"strictly from the supplied images. Answer: {answer}. Return {{claims:[string],claim_support:[boolean]}} "
                               "with one verdict per claim in the same order.",
                               images, FaithfulnessResponse)["claim_support"]
            row["visual_claim_faithfulness_ratio"] = sum(faith) / len(faith) if faith else None
            row.update(_ragas_scores(cfg, query, {"generated_answer": answer}, images))
        else:
            row.update({"visual_context_recall": 0.0, "visual_claim_faithfulness_ratio": 0.0,
                        "ragas_multimodal_faithfulness_binary": None, "ragas_answer_relevancy": None,
                        "native_metric_note": "not invoked with an empty visual context"})
        result["cases"][name] = row
        write_json(path, result)
    right, wrong, missing = (result["cases"][k] for k in cases)
    result["passed"] = (right.get("visual_context_recall", 0) >= 0.5 and
                        right.get("ragas_multimodal_faithfulness_binary") == 1 and
                        wrong.get("ragas_multimodal_faithfulness_binary") == 0 and
                        missing["visual_context_recall"] == 0 and right.get("ragas_answer_relevancy") is not None)
    write_json(path, result)
    if not result["passed"]:
        raise RuntimeError("real-image metric smoke checks failed; inspect eval/metric_smoke.json")
    return {"passed": True, "report": str(path)}

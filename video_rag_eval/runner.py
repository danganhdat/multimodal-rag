"""Run E1–E8 on the same frozen benchmark and record actual model context."""

from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path

from PIL import Image

from .core import append_jsonl, configured_api_key, frozen_benchmark, output, read_jsonl, source
from .corpus import load_maps
from .labels import VisionClient, _crop, _milvus
from .models import AnswerResponse, CropResponse, QueryResponse, SummaryResponse

EXPERIMENTS = {
    "E1": "SigLIP2 dense baseline",
    "E2": "query variants with RRF",
    "E3": "E2 plus object soft boost",
    "E4": "E2 plus SuperGlobal rerank",
    "E5": "E2 plus runtime visual crops",
    "E6": "E2 plus adjacent temporal evidence",
    "E7": "E2 plus generated shot summaries",
    "E8": "E2 seeded ReAct agent with inspect tools",
}


def _agent_usage_callback(ledger, model: str):
    from langchain_core.callbacks import BaseCallbackHandler

    class AgentUsage(BaseCallbackHandler):
        raise_error = True

        def on_chat_model_start(self, serialized, messages, **kwargs):
            ledger.check()

        def on_llm_start(self, serialized, prompts, **kwargs):
            ledger.check()

        def on_llm_end(self, response, **kwargs):
            usage = (response.llm_output or {}).get("token_usage", {})
            if not usage:
                metadata = getattr(response.generations[0][0].message, "usage_metadata", {}) or {}
                usage = {"prompt_tokens": metadata.get("input_tokens", 0),
                         "completion_tokens": metadata.get("output_tokens", 0)}
            ledger.record("E8_agent", usage, model)

    return AgentUsage()


def _query_variants(cfg: dict, client: VisionClient, q: dict) -> dict:
    path = output(cfg, "derived", "run_queries.jsonl")
    cached = {r["query_id"]: r for r in read_jsonl(path)}
    if q["query_id"] in cached:
        return cached[q["query_id"]]
    response = client.ask("run_query", "Translate scene description to English for visual search. "
                          "Make two independently worded visual variants. Extract visible object nouns only. "
                          f"Description: {q['description_vi']}. Return {{queries_en:[string,string,string],objects:[string]}}.", [], QueryResponse)
    variants = response.get("queries_en", [])
    if not variants or not all(isinstance(x, str) and x.strip() for x in variants):
        raise ValueError("query translation failed")
    result = {"query_id": q["query_id"], "queries_en": variants[:3],
              "objects": response.get("objects", [])}
    append_jsonl(path, result)
    return result


def _rrf(lists: list[list[dict]], k: int = 60) -> list[dict]:
    combined = {}
    for list_idx, hits in enumerate(lists):
        for rank, hit in enumerate(hits, 1):
            fid = hit["id"]
            if fid not in combined:
                combined[fid] = {**hit, "rrf_score": 0.0, "sources": {}}
            combined[fid]["rrf_score"] += 1 / (k + rank)
            combined[fid]["sources"][f"dense_{list_idx}"] = rank
    return sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)


def _image(cfg: dict, fid: str) -> Path:
    video, n = fid.split("/")
    return source(cfg, "keyframes") / video / f"{n}.jpg"


def _format_hit(hit: dict) -> dict:
    return {"frame_id": hit["id"], "video_id": hit.get("video_name", hit["id"].split("/")[0]),
            "frame_idx": int(hit["frame_idx"]),
            "score": float(hit.get("sources", {}).get("rerank", hit.get("rrf_score", hit.get("score", 0))))}


def _build_context(cfg: dict, experiment: str, hits: list[dict], q: dict, client: VisionClient) -> tuple[list[dict], list[Path], list[str]]:
    k = cfg["evaluation"]["context_k"]
    selected = hits[:k]
    paths = [_image(cfg, h["id"]) for h in selected]
    summaries = []
    if experiment == "E5":
        for hit in selected:
            path = _image(cfg, hit["id"])
            response = client.ask("runtime_crop", "Locate the image region most useful for answering: " + q["question_vi"] +
                                  ". Return {bbox:[x1,y1,x2,y2]} normalized to 0..1, or bbox=null if no useful crop.", [path], CropResponse)
            if isinstance(response.get("bbox"), list):
                dest = output(cfg, "runs", "crops", q["query_id"], hit["id"].replace("/", "_") + ".jpg")
                crop = _crop(path, response["bbox"], dest)
                if crop:
                    paths.append(crop)
                    summaries.append(f"Image {len(paths)} is a detail crop of frame {hit['id']}.")
    if experiment == "E6" and selected:
        maps, _ = load_maps(source(cfg, "maps"))
        head = selected[0]
        video, n = head["id"].split("/")
        number = int(n)
        for neighbor in (number - 1, number + 1):
            if neighbor in maps.get(video, {}):
                fid = f"{video}/{neighbor:03d}"
                if all(h["id"] != fid for h in selected):
                    meta = maps[video][neighbor]
                    selected.append({"id": fid, "video_name": video, "frame_idx": meta["frame_idx"], "score": 0})
                    paths.append(_image(cfg, fid))
        selected.sort(key=lambda h: (h.get("video_name", ""), h["frame_idx"]))
        paths = [_image(cfg, h["id"]) for h in selected]
    if experiment == "E7":
        for hit in selected:
            response = client.ask("shot_summary", "Describe only visible facts in this frame in one Vietnamese sentence. "
                                  "Return {summary:string}.", [_image(cfg, hit["id"])], SummaryResponse)
            summaries.append(response.get("summary", ""))
    return [_format_hit(h) for h in selected], paths, summaries


def _answer(client: VisionClient, q: dict, paths: list[Path], summaries: list[str], context_ids: list[str]) -> dict:
    prompt = ("Answer the Vietnamese question using only the attached images and any supplied summaries. "
              "If evidence is insufficient, answer status=uncertain and do not guess. "
              "Cite only frame IDs from the provided ordered list. "
              f"Scene description: {q['description_vi']} Question: {q['question_vi']}. "
              f"Frame IDs in order: {context_ids}. Extra images, if any, are detail crops of these frames. "
              f"Summaries: {json.dumps(summaries, ensure_ascii=False)}. "
              "Return {status:answered|uncertain|not_found,answer:string,citations:[frame_id]}.")
    return client.ask("run_answer", prompt, paths, AnswerResponse)


def run_experiment(cfg: dict, experiment: str, split: str = "test", services: dict | None = None) -> dict:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {experiment}")
    verification = output(cfg, "derived", "index_verification.json")
    if not verification.exists() or not json.loads(verification.read_text(encoding="utf-8")).get("passed"):
        raise RuntimeError("SigLIP2 index verification required before runs")
    import os
    os.environ.setdefault("LLM_MODEL", cfg["model"])
    os.environ.setdefault("VLM_MODEL", cfg["model"])
    # The existing backend expects LLM_API_KEY even when this CLI was given OPENROUTER_API_KEY.
    if not os.getenv("LLM_API_KEY"):
        os.environ["LLM_API_KEY"] = configured_api_key() or ""
    from backend.app.encoder import SigLIP2Encoder
    from backend.app.rerank.superglobal import rerank_superglobal
    benchmark = frozen_benchmark(cfg)
    queries = [r for r in read_jsonl(benchmark / "queries.jsonl") if r["split"] == split]
    if not queries:
        raise RuntimeError(f"no queries in {split} split")
    path = output(cfg, "runs", experiment + ".jsonl")
    done = {r["query_id"] for r in read_jsonl(path)}
    services = services if services is not None else {}
    if not services:
        services.update({"milvus": _milvus(cfg), "encoder": SigLIP2Encoder(cfg["milvus"]["siglip2_model"]),
                         "client": VisionClient(cfg)})
    milvus, encoder, client = services["milvus"], services["encoder"], services["client"]
    encoder._load()
    for q in queries:
        if q["query_id"] in done:
            continue
        start = time.perf_counter()
        start_cost = client.ledger.total
        translated = _query_variants(cfg, client, q)
        variants = translated["queries_en"] if experiment != "E1" else translated["queries_en"][:1]
        vectors = encoder.encode_texts(variants)
        lists = milvus.search_batch(vectors, top_k=300)
        hits = _rrf(lists) if experiment != "E1" else [{**h, "rrf_score": h["score"]} for h in lists[0]]
        if experiment == "E3":
            objects = {str(x).casefold() for x in translated.get("objects", [])}
            for hit in hits:
                if objects.intersection(str(x).casefold() for x in hit.get("objects", [])):
                    hit["rrf_score"] *= 1.15
            hits.sort(key=lambda x: x["rrf_score"], reverse=True)
        if experiment == "E4":
            hits = rerank_superglobal(hits, vectors[0], milvus, top=50)
        initial = [_format_hit(h) for h in hits[:300]]
        if experiment == "E8":
            from backend.app.agent.react import run_agent
            result = asyncio.run(run_agent(question=q["description_vi"] + "\nQuestion: " + q["question_vi"],
                                           milvus=milvus, siglip2=encoder, bm25=None, mode="agentic",
                                           candidates=[h["id"] for h in hits[:cfg["evaluation"]["context_k"]]], timeout_s=300,
                                           max_steps=cfg.get("agent_max_steps", 10),
                                           callbacks=[_agent_usage_callback(client.ledger, cfg["model"])]))
            context_ids = result.get("context_frame_ids", [])
            maps, _ = load_maps(source(cfg, "maps"))
            context = []
            for fid in context_ids:
                video, n = fid.split("/")
                if int(n) in maps.get(video, {}):
                    context.append({"frame_id": fid, "video_id": video,
                                    "frame_idx": maps[video][int(n)]["frame_idx"], "score": 0})
            answer = {"status": result.get("status"), "answer": result.get("answer", ""),
                      "citations": result.get("evidence", [])}
            context_paths = [Path(p) for p in result.get("context_image_paths", [])]
            context_texts = result.get("context_texts", [])
        else:
            context, paths, summaries = _build_context(cfg, experiment, hits, q, client)
            answer = _answer(client, q, paths, summaries, [h["frame_id"] for h in context])
            context_paths = paths
            context_texts = summaries
        record = {"query_id": q["query_id"], "config_id": experiment, "retrieved_hits": initial,
                  "context_hits": context, "context_frame_ids": [h["frame_id"] for h in context],
                  "context_image_paths": [str(p) for p in context_paths], "context_texts": context_texts,
                  "generated_answer": answer.get("answer", ""), "citations": answer.get("citations", []),
                  "status": answer.get("status", "uncertain"),
                  "latency_ms": (time.perf_counter() - start) * 1000,
                  "estimated_usd_delta": client.ledger.total - start_cost,
                  "estimated_usd_cumulative": client.ledger.total}
        append_jsonl(path, record)
    return {"experiment": experiment, "completed": len(read_jsonl(path)), "target": len(queries),
            "estimated_usd": client.ledger.total}


def run_all(cfg: dict, split: str = "test") -> dict:
    services = {}
    results = []
    for experiment in EXPERIMENTS:
        result = run_experiment(cfg, experiment, split, services)
        print(json.dumps(result), flush=True)
        results.append(result)
    return {"experiments": results}

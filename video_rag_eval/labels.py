"""Vision-model label generation, independent passes, qrel pooling and freeze."""

from __future__ import annotations

import base64
import csv
import io
import json
import random
import re
import time
from datetime import datetime, timezone
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from PIL import Image
from pydantic import BaseModel, ValidationError

from .core import BudgetExceeded, CostLedger, api_key, frozen_benchmark, output, read_jsonl, sha256, source, ssl_verify, write_json, write_jsonl, append_jsonl
from .corpus import load_maps
from .ranges import make_ranges


def _image_url(path: Path, max_side: int = 1024) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class VisionClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.key = api_key()
        self.ledger = CostLedger(cfg)
        self.model = cfg["model"]
        try:
            with httpx.Client(timeout=20, verify=ssl_verify()) as client:
                response = client.get(cfg["base_url"].rstrip("/") + "/key",
                                      headers={"Authorization": f"Bearer {self.key}"})
        except httpx.HTTPError as error:
            write_json(output(cfg, "derived", "api_preflight.json"),
                       {"usable": False, "reason": type(error).__name__})
            raise
        metadata = response.json().get("data", {}) if response.is_success else {}
        usable = response.is_success and not (metadata.get("is_management_key") or metadata.get("is_provisioning_key"))
        write_json(output(cfg, "derived", "api_preflight.json"),
                   {"usable": usable, "http_status": response.status_code,
                    "management_or_provisioning": bool(metadata.get("is_management_key") or metadata.get("is_provisioning_key"))})
        response.raise_for_status()
        if not usable:
            raise RuntimeError("OpenRouter key is management/provisioning only; configure an inference API key")

    def ask(self, stage: str, prompt: str, image_paths: list[Path], schema: type[BaseModel] | None = None) -> dict:
        self.ledger.check()
        content = [{"type": "text", "text": prompt + "\nReturn only a JSON object."}]
        content.extend({"type": "image_url", "image_url": {"url": _image_url(path)}} for path in image_paths)
        payload = {"model": self.model, "temperature": 0, "max_tokens": 3000,
                   "reasoning": {"max_tokens": self.cfg.get("reasoning_max_tokens", 512)},
                   "response_format": {"type": "json_object"},
                   "messages": [{"role": "user", "content": content}]}
        for attempt in range(5):
            self.ledger.check()
            try:
                with httpx.Client(timeout=90, verify=ssl_verify()) as client:
                    response = client.post(self.cfg["base_url"].rstrip("/") + "/chat/completions",
                                           headers={"Authorization": f"Bearer {self.key}"}, json=payload)
                response.raise_for_status()
                body = response.json()
                self.ledger.record(stage, body.get("usage", {}), self.model)
                if body["choices"][0].get("finish_reason") == "length":
                    raise ValueError("model output truncated at max_tokens")
                raw = body["choices"][0]["message"]["content"]
                result = json.loads(raw)
                if not isinstance(result, dict):
                    raise ValueError("model returned non-object JSON")
                if schema is not None:
                    result = schema.model_validate(result).model_dump()
                return result
            except (httpx.HTTPError, KeyError, ValueError, json.JSONDecodeError, ValidationError) as error:
                if isinstance(error, httpx.HTTPStatusError) and error.response.status_code in (400, 401, 402, 403):
                    raise
                if attempt == 4:
                    raise
                retry_after = error.response.headers.get("Retry-After") if isinstance(error, httpx.HTTPStatusError) else None
                try:
                    delay = min(30.0, max(2.0, float(retry_after))) if retry_after else min(30.0, 2 ** (attempt + 1))
                except ValueError:
                    delay = min(30.0, 2 ** (attempt + 1))
                if "truncated" in str(error):
                    payload["max_tokens"] = min(6000, payload["max_tokens"] + 1500)
                with self.ledger.lock:
                    append_jsonl(output(self.cfg, "derived", "api_events.jsonl"),
                                 {"stage": stage, "attempt": attempt + 1, "error_type": type(error).__name__,
                                  "http_status": error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None,
                                  "retry_wait_s": delay})
                time.sleep(delay)
        raise AssertionError("unreachable")


def _raise_systemic_api_error(error: Exception) -> None:
    """Do not turn network/auth failures affecting every frame into local rejects."""
    if isinstance(error, httpx.HTTPStatusError):
        if error.response.status_code != 400:
            raise error
    elif isinstance(error, (httpx.RequestError, OSError)):
        raise error


def _crop(image: Path, bbox: list[float], dest: Path) -> Path | None:
    if len(bbox) != 4 or any(not 0 <= float(x) <= 1 for x in bbox):
        return None
    with Image.open(image) as im:
        w, h = im.size
        x1, y1, x2, y2 = bbox
        box = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
        if box[2] - box[0] < 16 or box[3] - box[1] < 12:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.convert("RGB").crop(box).save(dest, quality=95)
    return dest


def extract_facts(cfg: dict, max_items: int | None = None) -> dict:
    from .models import FactResponse
    rows = read_jsonl(output(cfg, "derived", "sampled_frames.jsonl"))
    if not rows:
        raise RuntimeError("sampled_frames.jsonl missing; run sample")
    path = output(cfg, "derived", "frame_facts.jsonl")
    done = {r["frame_id"] for r in read_jsonl(path)}
    client = VisionClient(cfg)
    attempted = 0
    for row in rows:
        if row["frame_id"] in done:
            continue
        if max_items is not None and attempted >= max_items:
            break
        attempted += 1
        prompt = ("Inspect only visible pixels. Extract at most six grounded visual facts in Vietnamese. "
                  "Never infer text or place names from world knowledge. For unreadable text, transcription=null. "
                  "Return {visual_facts:[{fact_id,type,claim,transcription,confidence,bbox}], "
                  "uncertainties:[],usable_question_types:[]}. bbox is [x1,y1,x2,y2] normalized to 0..1.")
        try:
            fact = client.ask("facts", prompt, [Path(row["image_path"])], FactResponse)
            crops = []
            for i, item in enumerate(fact["visual_facts"]):
                if item.get("type") == "text" and isinstance(item.get("bbox"), list):
                    dest = output(cfg, "derived", "crops", row["video_id"], f"{row['n']:03d}_{i}.jpg")
                    crop = _crop(Path(row["image_path"]), item["bbox"], dest)
                    if crop:
                        crops.append(str(crop))
            append_jsonl(path, {"frame_id": row["frame_id"], "facts": fact,
                                "crop_paths": crops, "image_path": row["image_path"]})
        except BudgetExceeded:
            raise
        except Exception as e:
            _raise_systemic_api_error(e)
            append_jsonl(output(cfg, "derived", "quarantine.jsonl"),
                         {"frame_id": row["frame_id"], "stage": "facts", "reason": str(e)})
    return {"facts": len(read_jsonl(path)), "estimated_usd": client.ledger.total}


def generate_qa(cfg: dict) -> dict:
    from .models import QAResponse
    facts = read_jsonl(output(cfg, "derived", "frame_facts.jsonl"))
    path = output(cfg, "derived", "proposals.jsonl")
    done = {r["frame_id"] for r in read_jsonl(path)}
    client = VisionClient(cfg)
    quotas = cfg["sampling"]["quotas"]
    types = [kind for kind, fraction in quotas.items() for _ in range(round(20 * fraction))]
    random.Random(cfg["seed"]).shuffle(types)
    maps, _ = load_maps(source(cfg, "maps"))
    for i, row in enumerate(facts):
        if row["frame_id"] in done:
            continue
        kind = types[i % len(types)]
        images = [Path(row["image_path"])]
        source_ids = [row["frame_id"]]
        if kind == "multi_frame_temporal":
            video, n = row["frame_id"].split("/")
            mapping = maps.get(video, {})
            later = next((other for other in sorted(mapping) if other > int(n) and
                          mapping[other]["pts_time"] - mapping[int(n)]["pts_time"] >= 5), None)
            if later is not None:
                later_path = source(cfg, "keyframes") / video / f"{later:03d}.jpg"
                if later_path.is_file():
                    images.append(later_path)
                    source_ids.append(f"{video}/{later:03d}")
            if len(source_ids) < 2:
                kind = "object_attribute"
        prompt = (f"Generate one Vietnamese visual-search QA of type {kind} from this image and these facts: "
                  f"{json.dumps(row['facts'], ensure_ascii=False)}. "
                  "description_vi must be 2-5 natural sentences identifying the scene without leaking the answer. "
                  "question_vi must ask about pixels, not external knowledge or audio. "
                  "For OCR, use only legible crop pixels. If two ordered source frames are supplied, ask about a visible change between them. "
                  f"Ordered source frame IDs: {source_ids}. Return {{description_vi,question_vi,reference_answer," 
                  "retrieval_query_en,question_type,required_claims:[string],answerable:boolean}. "
                  "For unanswerable, reference_answer empty and required_claims empty.")
        images += [Path(p) for p in row.get("crop_paths", [])[:2]]
        try:
            qa = client.ask("generate", prompt, images, QAResponse)
            _resolve_claims(qa, row["facts"])
            append_jsonl(path, {"frame_id": row["frame_id"], "proposal": qa,
                                "crop_paths": row.get("crop_paths", []), "image_path": row["image_path"],
                                "source_frame_ids": source_ids,
                                "source_image_paths": [str(p) for p in images[:len(source_ids)]]})
        except BudgetExceeded:
            raise
        except Exception as e:
            _raise_systemic_api_error(e)
            append_jsonl(output(cfg, "derived", "quarantine.jsonl"),
                         {"frame_id": row["frame_id"], "stage": "generate", "reason": str(e)})
    return {"proposals": len(read_jsonl(path)), "estimated_usd": client.ledger.total}


def _norm(value: str) -> str:
    return re.sub(r"\W+", "", value.casefold(), flags=re.UNICODE)


def quality_failures(row: dict) -> list[str]:
    qa = row["proposal"]
    reasons = set()
    for vote in row.get("votes", []):
        for field in ("description_grounded", "question_unambiguous"):
            if vote.get(field) is not True:
                reasons.add(field)
        for field in ("leaks_answer",) + (("needs_more_pixels",) if qa["answerable"] else ()):
            if vote.get(field) is not False:
                reasons.add(field)
        if vote.get("answerable") != qa["answerable"]:
            reasons.add("answerability_mismatch")
    if qa["question_type"] == "ocr_signage" and not row.get("crop_paths"):
        reasons.add("ocr_without_crop")
    if qa["question_type"] == "multi_frame_temporal" and len(row.get("source_frame_ids", [])) < 2:
        reasons.add("temporal_missing_frames")
    if row.get("semantic_comparison", {}).get("equivalent") is False:
        reasons.add("reference_answer_mismatch")
    if row.get("claim_verification", {}).get("passed") is False:
        reasons.add("answer_claim_not_visually_supported")
    return sorted(reasons)


def _resolve_claims(qa: dict, facts: dict) -> dict:
    if not qa.get("answerable") and qa.get("question_type") != "unanswerable":
        qa["raw_question_type"] = qa.get("question_type")
        qa["question_type"] = "unanswerable"
    visual = facts.get("visual_facts", [])
    by_id = {str(f["fact_id"]).casefold(): f["claim"] for f in visual}
    by_number = {re.sub(r"\D", "", str(f["fact_id"])): f["claim"] for f in visual}
    resolved = []
    for claim in qa.get("required_claims", []):
        key = str(claim).strip().casefold()
        if key in by_id:
            resolved.append(by_id[key])
        elif re.fullmatch(r"(?:f|fact)[_ -]?\d+|\d+", key):
            number = re.sub(r"\D", "", key)
            if number not in by_number:
                raise ValueError(f"unresolved required claim reference: {claim}")
            resolved.append(by_number[number])
        else:
            resolved.append(claim)
    if resolved != qa.get("required_claims", []):
        qa["required_claim_refs"] = qa["required_claims"]
        qa["required_claims"] = resolved
    return qa


def normalize_label_claims(cfg: dict) -> None:
    facts = {r["frame_id"]: r["facts"] for r in read_jsonl(output(cfg, "derived", "frame_facts.jsonl"))}
    for name in ("proposals.jsonl", "judge_votes.jsonl"):
        path = output(cfg, "derived", name)
        rows = read_jsonl(path)
        changed = False
        for row in rows:
            before_type = row["proposal"].get("question_type")
            before = list(row["proposal"].get("required_claims", []))
            _resolve_claims(row["proposal"], facts.get(row["frame_id"], {}))
            changed = changed or before != row["proposal"].get("required_claims", []) or before_type != row["proposal"].get("question_type")
        if changed:
            write_jsonl(path, rows)


def _answers_equivalent(client: VisionClient, reference: str, answers: list[str], stage: str, question: str = "") -> dict:
    """Compare text after blind visual judgments; do not reveal reference to image judges."""
    from .models import EquivalenceResponse

    if all(_norm(str(answer)) == _norm(reference) for answer in answers):
        return {"equivalent": True, "reason": "normalized_exact_match"}
    prompt = ("Compare Vietnamese answers to a reference relative to the question. Each answer must "
              "give the same essential answer to that question without a contradictory fact. Accept paraphrases "
              "and omission of ancillary details that the question does not ask for. Reject a wrong object, "
              "number, action or temporal order that the question asks about. "
              f"Question: {question}. Reference: {reference}. Answers: {json.dumps(answers, ensure_ascii=False)}. "
              "Return {equivalent:boolean,reason:string}.")
    return client.ask(stage, prompt, [], EquivalenceResponse)


def _judge_qrel(client: VisionClient, qa: dict, images: list[Path], stage: str = "qrel") -> dict:
    from .models import QrelResponse

    claims = [{"claim_id": i, "claim": claim} for i, claim in enumerate(qa.get("required_claims", []), 1)]
    prompt = ("Judge visual relevance of this candidate frame (and detail crops of the same frame). Ignore retriever scores. "
              f"Description: {qa['description_vi']} Question: {qa['question_vi']}. "
              f"Reference visual claims: {json.dumps(claims, ensure_ascii=False)}. "
              "Grades: 3=directly answers from visible pixels, 2=almost sufficient or confidently supports one required event "
              "in a multi-frame temporal question, 1=topic only, 0=irrelevant. A temporal frame may be grade 2 even though "
              "the full temporal answer needs another frame. uncertain=true means the RELEVANCE judgment itself is unclear; "
              "do not set it merely because the candidate cannot answer the entire question. Never guess unreadable OCR. "
              "Return {relevance:0..3,derived_answer:string,supports_claim_ids:[integer],reason_code:string,uncertain:boolean}.")
    vote = client.ask(stage, prompt, images, QrelResponse)
    grade = int(vote["relevance"])
    second = client.ask(stage + "_confirm", prompt + " Make a separate blind visual judgment.", images, QrelResponse) if grade == 3 else None
    consensus = not vote["uncertain"] and (grade != 3 or (second is not None and not second["uncertain"] and second["relevance"] == 3))
    comparison = None
    if consensus and grade == 3 and qa["answerable"]:
        comparison = _answers_equivalent(client, qa["reference_answer"],
                                         [vote["derived_answer"], second["derived_answer"]],
                                         stage + "_answer_equivalence", qa["question_vi"])
    return {"status": "judged" if consensus and (comparison is None or comparison["equivalent"]) else "unjudged_disagreement",
            "grade": grade, "vote": vote, "second_vote": second, "answer_comparison": comparison,
            "answer_comparison_version": "question_aware_v2", "qrel_protocol": "claim_aware_v3",
            "judged_claims": qa.get("required_claims", [])}


def verify_qa(cfg: dict) -> dict:
    from .models import VerifyResponse
    normalize_label_claims(cfg)
    proposals = read_jsonl(output(cfg, "derived", "proposals.jsonl"))
    path = output(cfg, "derived", "judge_votes.jsonl")
    client = VisionClient(cfg)

    def accepted_vote(row: dict) -> dict:
        qa, votes = row["proposal"], row["votes"]
        visual_ok = len(votes) == 2 and all(
            isinstance(v.get("reasons"), list) and v.get("description_grounded") is True and
            v.get("question_unambiguous") is True and v.get("leaks_answer") is False and
            (not qa["answerable"] or v.get("needs_more_pixels") is False) and v.get("answerable") == qa["answerable"]
            for v in votes)
        if qa["question_type"] == "ocr_signage" and not row.get("crop_paths"):
            visual_ok = False
        if qa["question_type"] == "multi_frame_temporal" and len(row.get("source_frame_ids", [])) < 2:
            visual_ok = False
        equivalent = True
        if visual_ok and qa["answerable"]:
            equivalent = all(_norm(str(v.get("independent_answer", ""))) == _norm(qa["reference_answer"]) for v in votes)
            if not equivalent:
                if row.get("semantic_comparison_version") != "question_aware_v2":
                    row["semantic_comparison"] = _answers_equivalent(
                        client, qa["reference_answer"],
                        [v.get("independent_answer", "") for v in votes], "answer_equivalence", qa["question_vi"])
                    row["semantic_comparison_version"] = "question_aware_v2"
                equivalent = row["semantic_comparison"]["equivalent"] is True
        row["accepted"] = bool(visual_ok and equivalent and row.get("claim_verification", {}).get("passed", True))
        row["reason"] = "consensus" if row["accepted"] else "judge_disagreement_or_quality_gate"
        row["quality_gate_failures"] = quality_failures(row)
        return row

    existing = read_jsonl(path)
    changed = False
    for row in existing:
        if not row.get("accepted") and row.get("votes"):
            before = row["accepted"]
            accepted_vote(row)
            changed = changed or row["accepted"] != before or "semantic_comparison" in row
    if changed:
        write_jsonl(path, existing)
    done = {r["frame_id"] for r in existing}
    for row in proposals:
        if row["frame_id"] in done:
            continue
        qa = row["proposal"]
        prompt = ("Judge this Vietnamese QA solely from the supplied image. First answer independently; "
                  "you must not guess unreadable text. Check grounding, ambiguity and answer leakage. "
                  f"Description: {qa['description_vi']} Question: {qa['question_vi']}. "
                  "Return {independent_answer,answerable,description_grounded,question_unambiguous," 
                  "leaks_answer,needs_more_pixels,reasons:[string]}.")
        images = [Path(p) for p in row.get("source_image_paths", [row["image_path"]])] + [Path(p) for p in row.get("crop_paths", [])[:2]]
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(client.ask, f"verify_{j}",
                                           prompt + f" Judge pass {j}; make an independent assessment.",
                                           images, VerifyResponse) for j in ("a", "b")]
                votes = [future.result() for future in futures]
            vote_row = {"frame_id": row["frame_id"], "proposal": qa, "votes": votes,
                        "image_path": row["image_path"], "crop_paths": row.get("crop_paths", []),
                        "source_frame_ids": row.get("source_frame_ids", [row["frame_id"]])}
            append_jsonl(path, accepted_vote(vote_row))
        except BudgetExceeded:
            raise
        except Exception as e:
            _raise_systemic_api_error(e)
            append_jsonl(output(cfg, "derived", "quarantine.jsonl"),
                         {"frame_id": row["frame_id"], "stage": "verify", "reason": str(e)})
    votes = read_jsonl(path)
    return {"accepted": sum(r["accepted"] for r in votes), "rejected": sum(not r["accepted"] for r in votes),
            "estimated_usd": client.ledger.total}


def _milvus(cfg: dict):
    from backend.app.milvus_client import MilvusSearchService
    return MilvusSearchService(cfg["milvus"]["uri"], cfg["milvus"]["collection"])


def select_verified(cfg: dict, votes: list[dict]) -> list[dict]:
    unique = {}
    for row in votes:
        if row.get("accepted"):
            qa = row["proposal"]
            unique.setdefault(_norm(qa["description_vi"] + qa["question_vi"]), row)
    candidates = list(unique.values())
    selected = set()
    for kind, fraction in cfg["sampling"]["quotas"].items():
        count = round(cfg["target_qa"] * fraction)
        matches = [r for r in candidates if r["proposal"]["question_type"] == kind]
        selected.update(r["frame_id"] for r in matches[:min(count, cfg["target_qa"] - len(selected))])
    for row in candidates:
        if len(selected) >= cfg["target_qa"]:
            break
        selected.add(row["frame_id"])
    return [r for r in candidates if r["frame_id"] in selected][:cfg["target_qa"]]


def verify_claims(cfg: dict) -> dict:
    """Decompose essential answer claims and verify the entire ordered source evidence."""
    from .models import ClaimsResponse, SupportResponse

    normalize_label_claims(cfg)
    votes_path = output(cfg, "derived", "judge_votes.jsonl")
    votes = read_jsonl(votes_path)
    accepted = [r for r in votes if r.get("accepted")]
    path = output(cfg, "derived", "claim_verification.jsonl")
    cached = {r["frame_id"]: r for r in read_jsonl(path)}
    client = VisionClient(cfg)

    def judge_claims(row):
        fid, qa = row["frame_id"], row["proposal"]
        if not qa["answerable"]:
            return {"frame_id": fid, "claims": [], "supported": [], "passed": True, "protocol": "answer_claims_v2"}
        decomposition = client.ask("answer_claim_decomposition",
            "Split the reference answer into 1-4 atomic visual claims essential to answering the question. "
            "Write explicit Vietnamese assertions, including exact requested text/numbers. Exclude ancillary scene facts. "
            "For a temporal answer include the first state/value, the second state/value, and their ordered change. "
            f"Question: {qa['question_vi']}. Reference answer: {qa['reference_answer']}. Return {{claims:[string]}}.",
            [], ClaimsResponse)
        ids = row.get("source_frame_ids", [fid])
        images = [source(cfg, "keyframes") / video / f"{n}.jpg" for video, n in (item.split("/") for item in ids)]
        images.extend(Path(p) for p in row.get("crop_paths", []) if Path(p).is_file())
        support = client.ask("answer_claim_verification",
            "For every claim judge whether the supplied ordered source images directly support the entire claim. "
            "Do not infer unreadable text or unseen events. A temporal relation needs both endpoint values/events "
            "and their order. If any essential part is not visible, mark that claim false. "
            f"Question: {qa['question_vi']}. First {len(ids)} images are source frames in this order: {ids}; "
            "remaining images are detail crops of the first frame. "
            f"Claims: {json.dumps(decomposition['claims'], ensure_ascii=False)}. Return {{supported:[boolean]}}.",
            images, SupportResponse)["supported"]
        if len(support) != len(decomposition["claims"]):
            raise ValueError("claim verifier returned the wrong number of verdicts")
        return {"frame_id": fid, "claims": decomposition["claims"], "supported": support,
                "passed": all(support), "protocol": "answer_claims_v2"}

    pending = [r for r in accepted if r["frame_id"] not in cached]
    with ThreadPoolExecutor(max_workers=cfg.get("api_workers", 2)) as executor:
        futures = [executor.submit(judge_claims, row) for row in pending]
        failure = None
        for future in as_completed(futures):
            if future.cancelled():
                continue
            try:
                result = future.result()
                append_jsonl(path, result)
                cached[result["frame_id"]] = result
            except Exception as error:
                failure = failure or error
                for other in futures:
                    other.cancel()
        if failure:
            raise failure
    previous_claims = {r["frame_id"]: list(r["proposal"].get("required_claims", [])) for r in votes}
    for row in votes:
        result = cached.get(row["frame_id"])
        if result is None:
            continue
        row["claim_verification"] = result
        row["proposal"].setdefault("previous_required_claims", row["proposal"].get("required_claims", []))
        row["proposal"]["required_claims"] = result["claims"]
        if not result["passed"]:
            row["accepted"] = False
            row["reason"] = "answer_claim_not_visually_supported"
    write_jsonl(votes_path, votes)
    proposals = read_jsonl(output(cfg, "derived", "proposals.jsonl"))
    by_id = {r["frame_id"]: r["proposal"] for r in votes}
    for row in proposals:
        if row["frame_id"] in by_id:
            row["proposal"] = by_id[row["frame_id"]]
    write_jsonl(output(cfg, "derived", "proposals.jsonl"), proposals)
    pool_path = output(cfg, "derived", "pool_judgments.jsonl")
    judgments = read_jsonl(pool_path)
    for row in judgments:
        row.setdefault("judged_claims", previous_claims.get(row["frame_id"], []))
    if judgments:
        write_jsonl(pool_path, judgments)
    remaining = sum(bool(r.get("accepted")) for r in votes)
    if remaining < cfg["target_qa"]:
        raise RuntimeError(f"only {remaining}/{cfg['target_qa']} QA retain visual support for all answer claims")
    return {"claim_verified": sum(r["passed"] for r in cached.values()), "accepted": remaining,
            "estimated_usd": client.ledger.total}


def pool_and_label(cfg: dict) -> dict:
    from backend.app.encoder import SigLIP2Encoder
    normalize_label_claims(cfg)
    verified = select_verified(cfg, read_jsonl(output(cfg, "derived", "judge_votes.jsonl")))
    if not verified:
        raise RuntimeError("no accepted QA")
    if not output(cfg, "derived", "index_verification.json").exists():
        raise RuntimeError("verify-index must pass before pooling")
    milvus = _milvus(cfg)
    encoder = SigLIP2Encoder(cfg["milvus"]["siglip2_model"])
    client = VisionClient(cfg)
    path = output(cfg, "derived", "pool_judgments.jsonl")
    previous = read_jsonl(path)
    qa_by_source = {r["frame_id"]: r["proposal"] for r in verified}
    source_rows = {r["frame_id"]: r for r in verified}
    reconciled = False
    for item in previous:
        source_row = source_rows.get(item["frame_id"])
        if (source_row and item.get("status") != "judged" and
            (item.get("qrel_protocol") != "claim_aware_v3" or item.get("source_retry_count", 0) < 1) and
            item["source_frame_id"] in source_row.get("source_frame_ids", [item["frame_id"]])):
            video, n = item["source_frame_id"].split("/")
            images = [source(cfg, "keyframes") / video / f"{n}.jpg"]
            if item["source_frame_id"] == item["frame_id"]:
                images.extend(Path(p) for p in source_row.get("crop_paths", []) if Path(p).is_file())
            item.setdefault("judgment_history", []).append({k: item.get(k) for k in
                ("status", "grade", "vote", "second_vote", "answer_comparison", "qrel_protocol")})
            item.update(_judge_qrel(client, source_row["proposal"], images))
            item["source_retry_count"] = item.get("source_retry_count", 0) + 1
            reconciled = True
        first, second = item.get("vote", {}), item.get("second_vote") or {}
        if (item.get("status") == "unjudged_disagreement" and item.get("grade") == 3 and
            first.get("uncertain") is False and second.get("uncertain") is False and second.get("relevance") == 3 and
            item.get("answer_comparison_version") != "question_aware_v2" and item["frame_id"] in qa_by_source):
            qa = qa_by_source[item["frame_id"]]
            item.setdefault("answer_comparison_history", []).append(item.get("answer_comparison"))
            item["answer_comparison"] = _answers_equivalent(
                client, qa["reference_answer"], [first.get("derived_answer", ""), second.get("derived_answer", "")],
                "qrel_answer_equivalence", qa["question_vi"])
            item["answer_comparison_version"] = "question_aware_v2"
            item["status"] = "judged" if item["answer_comparison"]["equivalent"] else "unjudged_disagreement"
            reconciled = True
    if reconciled:
        write_jsonl(path, previous)
    done = {(r["frame_id"], r["source_frame_id"]) for r in previous}
    pool_cfg = cfg["pool"]
    rng = random.Random(cfg["seed"])
    manifest = read_jsonl(output(cfg, "derived", "frames.jsonl"))
    all_ids = [r["frame_id"] for r in manifest]
    for row in verified[:cfg["target_qa"]]:
        source_id, qa = row["frame_id"], row["proposal"]
        query = qa["retrieval_query_en"]
        dense = milvus.search_single(encoder.encode_text(query), top_k=pool_cfg["dense_top"])
        source_meta = milvus.get_by_ids([source_id])
        objects = source_meta[0].get("objects", []) if source_meta else []
        auxiliary = []
        if objects:
            safe = str(objects[0]).replace('"', '')
            auxiliary = milvus.get_by_ids_expr(f'ARRAY_CONTAINS(objects, "{safe}")',
                                                output_fields=["id", "objects", "ocr"],
                                                limit=pool_cfg["auxiliary_top"])
        random_ids = rng.sample(all_ids, min(pool_cfg["random_negatives"], len(all_ids)))
        candidates = {fid: {"sources": ["source"]} for fid in row.get("source_frame_ids", [source_id])}
        for name, hits in (("siglip2", dense), ("object", auxiliary)):
            for rank, hit in enumerate(hits):
                fid = hit["id"]
                candidates.setdefault(fid, {"sources": []})["sources"].append(f"{name}:{rank}")
        for fid in random_ids:
            candidates.setdefault(fid, {"sources": []})["sources"].append("random")
        def judge_candidate(item):
            fid, meta = item
            image = source(cfg, "keyframes") / fid.split("/")[0] / (fid.split("/")[1] + ".jpg")
            if not image.is_file():
                return {"frame_id": source_id, "source_frame_id": fid,
                        "status": "unjudged_missing_image", **meta}
            images = [image]
            if fid == source_id:
                images.extend(Path(p) for p in row.get("crop_paths", []) if Path(p).is_file())
            try:
                return {"frame_id": source_id, "source_frame_id": fid,
                        **_judge_qrel(client, qa, images), **meta}
            except BudgetExceeded:
                raise
            except Exception as e:
                _raise_systemic_api_error(e)
                return {"frame_id": source_id, "source_frame_id": fid,
                        "status": "unjudged_error", "reason": str(e), **meta}
        pending = [(fid, meta) for fid, meta in candidates.items() if (source_id, fid) not in done]
        failure = None
        with ThreadPoolExecutor(max_workers=min(4, cfg.get("api_workers", 2))) as executor:
            futures = [executor.submit(judge_candidate, item) for item in pending]
            for future in as_completed(futures):
                if future.cancelled():
                    continue
                try:
                    result = future.result()
                    append_jsonl(path, result)
                    done.add((result["frame_id"], result["source_frame_id"]))
                except Exception as error:
                    failure = failure or error
                    for pending_future in futures:
                        pending_future.cancel()
        if failure:
            raise failure
    rows = read_jsonl(path)
    return {"judged": sum(r["status"] == "judged" for r in rows),
            "unjudged": sum(r["status"] != "judged" for r in rows), "estimated_usd": client.ledger.total}


def freeze(cfg: dict, allow_partial: bool = False) -> dict:
    manifest_path = output(cfg, "benchmark", "v1", "benchmark_manifest.json")
    if manifest_path.exists():
        benchmark = frozen_benchmark(cfg)
        manifest = json.loads((benchmark / "benchmark_manifest.json").read_text(encoding="utf-8"))
        if manifest["qa_count"] < cfg["target_qa"] and not allow_partial:
            raise RuntimeError("existing v1 is already frozen with fewer QA; choose a new data_root for a new benchmark")
        return {**manifest, "reused": True}
    normalize_label_claims(cfg)
    chosen = select_verified(cfg, read_jsonl(output(cfg, "derived", "judge_votes.jsonl")))
    if not chosen or (len(chosen) < cfg["target_qa"] and not allow_partial):
        raise RuntimeError(f"only {len(chosen)}/{cfg['target_qa']} QA verified; use --allow-partial to inspect")
    judgments = read_jsonl(output(cfg, "derived", "pool_judgments.jsonl"))
    by_source = defaultdict(list)
    for r in judgments:
        if r["status"] == "judged":
            by_source[r["frame_id"]].append(r)
    for row in chosen:
        if row["proposal"]["answerable"]:
            if row.get("claim_verification", {}).get("passed") is not True:
                raise RuntimeError(f"answer claim verification required before freeze: {row['frame_id']}; run claims")
            source_ids = row.get("source_frame_ids", [row["frame_id"]])
            minimum = 2 if len(source_ids) > 1 else 3
            if any(not any(r["source_frame_id"] == fid and r["grade"] >= minimum for r in by_source[row["frame_id"]]) for fid in source_ids):
                raise RuntimeError(f"source evidence lacks verified qrel: {row['frame_id']}")
        elif any(r["grade"] == 3 for r in by_source[row["frame_id"]]):
            raise RuntimeError(f"unanswerable QA has a directly answering frame in the pool: {row['frame_id']}")
    benchmark = output(cfg, "benchmark", "v1", "benchmark_manifest.json").parent
    rng = random.Random(cfg["seed"])
    videos = sorted({r["frame_id"].split("/")[0] for r in chosen})
    parent = {v: v for v in videos}
    def find(v):
        while parent[v] != v:
            v = parent[v]
        return v
    def union(a, b):
        parent[find(b)] = find(a)
    by_hash = defaultdict(set)
    for frame in read_jsonl(output(cfg, "derived", "frames.jsonl")):
        if frame["video_id"] in parent:
            by_hash[frame["sha256"]].add(frame["video_id"])
    for copies in by_hash.values():
        ordered_copies = sorted(copies)
        for other in ordered_copies[1:]:
            union(ordered_copies[0], other)
    groups = defaultdict(list)
    for video in videos:
        groups[find(video)].append(video)
    group_list = list(groups.values())
    rng.shuffle(group_list)
    dev_count = max(1, round(0.3 * len(group_list))) if len(group_list) > 1 else 0
    dev = {v for group in group_list[:dev_count] for v in group}
    queries, answers, evidence, qrels = [], [], [], []
    for i, row in enumerate(chosen, 1):
        qa, fid = row["proposal"], row["frame_id"]
        qid = f"Q{i:04d}"
        queries.append({"query_id": qid, "description_vi": qa["description_vi"],
                        "question_vi": qa["question_vi"], "question_type": qa["question_type"],
                        "answerable": qa["answerable"], "split": "dev" if fid.split("/")[0] in dev else "test"})
        answers.append({"query_id": qid, "reference_answer": qa["reference_answer"],
                        "required_claims": qa["required_claims"], "answerable": qa["answerable"],
                        "label_status": "verified_silver"})
        source_ids = row.get("source_frame_ids", [fid])
        evidence.append({"query_id": qid, "source_video_id": fid.split("/")[0], "source_frame_id": fid,
                         "temporal_relation": "ordered" if len(source_ids) > 1 else None,
                         "evidence_sets": [{"set_id": "ES1", "frame_ids": source_ids, "required_all": True,
                                            "crop_paths": row.get("crop_paths", [])}] if qa["answerable"] else [],
                         "judge_ids": ["qwen3.8_flash_pass_a", "qwen3.8_flash_pass_b"]})
        qrels.extend({"query_id": qid, "frame_id": r["source_frame_id"], "grade": r["grade"]}
                     for r in by_source[fid])
    for name, rows in (("queries.jsonl", queries), ("answers.jsonl", answers), ("evidence.jsonl", evidence)):
        write_jsonl(benchmark / name, rows)
    qrels.sort(key=lambda r: (r["query_id"], r["frame_id"]))
    with (benchmark / "qrels.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows((r["query_id"], 0, r["frame_id"], r["grade"]) for r in qrels)
    maps, _ = load_maps(source(cfg, "maps"))
    ordered = {v: [m[n]["frame_idx"] for n in sorted(m)] for v, m in maps.items()}
    ranges = make_ranges(qrels, ordered)
    with (benchmark / "accepted_ranges.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "video_id", "lo_frame_idx", "hi_frame_idx", "grade", "source_frame_id"], delimiter="\t")
        writer.writeheader()
        writer.writerows(ranges)
    write_json(benchmark / "splits.json", {"dev_videos": sorted(dev), "test_videos": sorted(set(videos) - dev),
                                           "duplicate_video_groups": list(groups.values())})
    files = [benchmark / p for p in ("queries.jsonl", "qrels.tsv", "answers.jsonl", "evidence.jsonl", "accepted_ranges.tsv", "splits.json")]
    manifest = {"version": "v1", "silver": True, "seed": cfg["seed"], "model": cfg["model"],
                "qa_count": len(queries), "qrel_count": len(qrels), "judge_independence": "same checkpoint, independent prompts",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "config_sha256": sha256(Path(cfg["config_path"])),
                "corpus_manifest_sha256": sha256(output(cfg, "derived", "frames.jsonl")) if output(cfg, "derived", "frames.jsonl").is_file() else None,
                "implementation_hashes": {name: sha256(Path(__file__).with_name(name)) for name in ("labels.py", "runner.py", "scoring.py")},
                "label_protocol": "blind_visual_passes_plus_question_aware_text_comparison_v2_and_answer_claims_v2",
                "ragas_version": cfg["evaluation"]["ragas_version"], "pool_config": cfg["pool"],
                "hashes": {p.name: sha256(p) for p in files}}
    write_json(benchmark / "benchmark_manifest.json", manifest)
    all_votes = read_jsonl(output(cfg, "derived", "judge_votes.jsonl"))
    write_json(benchmark / "label_quality_report.json", {
        "silver": True,
        "proposals": len(read_jsonl(output(cfg, "derived", "proposals.jsonl"))),
        "verified": len(all_votes),
        "accepted": sum(bool(r.get("accepted")) for r in all_votes),
        "rejected": sum(not r.get("accepted") for r in all_votes),
        "judge_disagreement_or_quality_rejection": sum(r.get("reason") == "judge_disagreement_or_quality_gate" for r in all_votes),
        "quality_gate_failures": dict(Counter(reason for r in all_votes if not r.get("accepted") for reason in quality_failures(r))),
        "judge_flag_disagreements": sum(len(r.get("votes", [])) == 2 and any(
            r["votes"][0].get(k) != r["votes"][1].get(k) for k in
            ("answerable", "description_grounded", "question_unambiguous", "leaks_answer", "needs_more_pixels")) for r in all_votes),
        "question_types": dict(Counter(r["question_type"] for r in queries)),
        "pool_judged": sum(r.get("status") == "judged" for r in judgments),
        "pool_unjudged": sum(r.get("status") != "judged" for r in judgments),
        "qrel_positive": sum(r["grade"] >= 2 for r in qrels),
        "qrel_negative": sum(r["grade"] == 0 for r in qrels),
    })
    return manifest


def adjudicate_runs(cfg: dict) -> dict:
    """Blindly judge the union of top run frames without rewriting frozen qrels."""
    from .runner import EXPERIMENTS
    benchmark = frozen_benchmark(cfg)
    queries = {r["query_id"]: r for r in read_jsonl(benchmark / "queries.jsonl")}
    answers = {r["query_id"]: r for r in read_jsonl(benchmark / "answers.jsonl")}
    frozen = set()
    with (benchmark / "qrels.tsv").open(newline="", encoding="utf-8") as f:
        for qid, _, fid, _ in csv.reader(f, delimiter="\t"):
            frozen.add((qid, fid))
    candidates = set()
    crops = defaultdict(set)
    for exp in EXPERIMENTS:
        for run in read_jsonl(output(cfg, "runs", exp + ".jsonl")):
            for hit in run.get("retrieved_hits", [])[:cfg["pool"]["supplemental_top"]] + run.get("context_hits", []):
                candidates.add((run["query_id"], hit["frame_id"]))
                prefix = hit["frame_id"].replace("/", "_")
                for image_path in run.get("context_image_paths", []):
                    p = Path(image_path)
                    if p.stem == prefix or p.stem.startswith(prefix + "_"):
                        crops[(run["query_id"], hit["frame_id"])].add(str(p))
    path = output(cfg, "eval", "supplemental_judgments.jsonl")
    done = {(r["query_id"], r["frame_id"]) for r in read_jsonl(path)} | frozen
    client = VisionClient(cfg)
    def judge_candidate(item):
        qid, fid = item
        if qid not in queries:
            raise ValueError(f"unknown query in runs: {qid}")
        video, n = fid.split("/")
        image = source(cfg, "keyframes") / video / f"{n}.jpg"
        if not image.is_file():
            return {"query_id": qid, "frame_id": fid, "status": "unjudged_missing_image"}
        images = [image] + [Path(p) for p in sorted(crops[item])[:2] if Path(p).is_file()]
        try:
            return {"query_id": qid, "frame_id": fid, "judged_image_paths": [str(p) for p in images],
                    **_judge_qrel(client, {**queries[qid], **answers[qid]}, images, "eval_qrel")}
        except BudgetExceeded:
            raise
        except Exception as e:
            _raise_systemic_api_error(e)
            return {"query_id": qid, "frame_id": fid, "status": "unjudged_error", "reason": str(e)}
    failure = None
    with ThreadPoolExecutor(max_workers=min(2, cfg.get("api_workers", 2))) as executor:
        futures = [executor.submit(judge_candidate, item) for item in sorted(candidates - done)]
        for future in as_completed(futures):
            if future.cancelled():
                continue
            try:
                append_jsonl(path, future.result())
            except Exception as error:
                failure = failure or error
                for other in futures:
                    other.cancel()
        if failure:
            raise failure
    judged = [r for r in read_jsonl(path) if r["status"] == "judged"]
    maps, _ = load_maps(source(cfg, "maps"))
    ordered = {v: [m[n]["frame_idx"] for n in sorted(m)] for v, m in maps.items()}
    ranges = make_ranges(judged, ordered)
    with output(cfg, "eval", "supplemental_ranges.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "video_id", "lo_frame_idx", "hi_frame_idx", "grade", "source_frame_id"], delimiter="\t")
        writer.writeheader()
        writer.writerows(ranges)
    return {"candidate_pairs": len(candidates), "new_judged": len(judged),
            "estimated_usd": client.ledger.total}

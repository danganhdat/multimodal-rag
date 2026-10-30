"""RAG evaluation metrics for Video QA — aligned with playbook C3/C4/C5.

Per-tier metrics:

  Retrieval tier:
    - recall_at_k             |S ∩ Z_k| / |S|  at configurable K
    - ndcg_at_k               nDCG with graded relevance (qrels 0-3)
    - mrr                     1 / rank of first relevant result
    - precision_at_k          |relevant in top-K| / K

  RAG context tier (on final evidence fed to MLLM):
    - context_recall          |S ∩ Z_k| / |S|  (C3)
    - context_precision       rank-aware AP     (C3)

  RAG answer tier (LLM-judge):
    - faithfulness            |C⁺| / |C|  claim-level  (C3)
    - answer_relevance        LLM-judged proxy 0-1      (proxy for C3 cosine formula)

  Supplementary:
    - answer_token_f1         token-level F1 vs ground truth

LLM-judge metrics return score=None when the LLM is unavailable.
Offline metrics return score=None only when ground truth is missing.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RetrievalSample:
    """One retrieval evaluation sample with ranked results and qrels."""
    query_id: str = ""
    retrieved_ids: list[str] = field(default_factory=list)
    qrels: dict[str, int] = field(default_factory=dict)
    relevance_threshold: int = 2


@dataclass
class EvalSample:
    """One RAG evaluation sample: question + answer + evidence + ground truth."""
    question: str
    answer: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    evidence_texts: list[str] = field(default_factory=list)
    ground_truth: str = ""
    ground_truth_ids: list[str] = field(default_factory=list)


@dataclass
class MetricResult:
    name: str
    score: float | None
    details: dict = field(default_factory=dict)


# ===========================================================================
# Retrieval-tier metrics (offline, qrels-based)
# ===========================================================================

def recall_at_k(sample: RetrievalSample, k: int) -> MetricResult:
    """Recall@K: fraction of relevant items in the top-K results.

    |S ∩ Z_k| / |S| where S = {id : qrels[id] >= threshold}.
    """
    relevant = {rid for rid, rel in sample.qrels.items() if rel >= sample.relevance_threshold}
    if not relevant:
        return MetricResult(f"recall@{k}", None, {"reason": "no_relevant_in_qrels"})

    top_k = set(sample.retrieved_ids[:k])
    hits = relevant & top_k
    score = len(hits) / len(relevant)
    return MetricResult(f"recall@{k}", round(score, 4), {
        "hits": len(hits), "n_relevant": len(relevant), "k": k,
    })


def precision_at_k(sample: RetrievalSample, k: int) -> MetricResult:
    """Precision@K: fraction of top-K that are relevant."""
    top_k = sample.retrieved_ids[:k]
    if not top_k:
        return MetricResult(f"precision@{k}", 0.0, {"k": k})

    relevant = {rid for rid, rel in sample.qrels.items() if rel >= sample.relevance_threshold}
    hits = sum(1 for rid in top_k if rid in relevant)
    score = hits / len(top_k)
    return MetricResult(f"precision@{k}", round(score, 4), {
        "hits": hits, "k": len(top_k),
    })


def ndcg_at_k(sample: RetrievalSample, k: int) -> MetricResult:
    """nDCG@K with graded relevance from qrels (0-3 scale)."""
    top_k = sample.retrieved_ids[:k]
    if not top_k:
        return MetricResult(f"ndcg@{k}", 0.0, {"k": k})

    dcg = sum(
        sample.qrels.get(rid, 0) / math.log2(i + 2)
        for i, rid in enumerate(top_k)
    )

    ideal_rels = sorted(sample.qrels.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    score = dcg / idcg if idcg > 0 else 0.0
    return MetricResult(f"ndcg@{k}", round(score, 4), {
        "dcg": round(dcg, 4), "idcg": round(idcg, 4), "k": k,
    })


def mrr(sample: RetrievalSample) -> MetricResult:
    """Mean Reciprocal Rank: 1 / rank of first relevant result."""
    relevant = {rid for rid, rel in sample.qrels.items() if rel >= sample.relevance_threshold}
    for i, rid in enumerate(sample.retrieved_ids):
        if rid in relevant:
            rank = i + 1
            return MetricResult("mrr", round(1.0 / rank, 4), {"first_relevant_rank": rank})
    return MetricResult("mrr", 0.0, {"first_relevant_rank": None})


# ===========================================================================
# RAG context-tier metrics (offline, on final evidence)
# ===========================================================================

def context_recall(sample: EvalSample) -> MetricResult:
    """C3 Context Recall: |S ∩ Z_k| / |S|.

    S = ground_truth_ids (relevant frames from ground truth).
    Z_k = evidence_ids (frames actually fed to MLLM).
    """
    if not sample.ground_truth_ids:
        return MetricResult("context_recall", None, {"reason": "no_gt_ids"})

    gt = set(sample.ground_truth_ids)
    retrieved = set(sample.evidence_ids)
    hits = gt & retrieved
    score = len(hits) / len(gt)

    return MetricResult("context_recall", round(score, 4), {
        "hits": sorted(hits),
        "missed": sorted(gt - retrieved),
        "n_gt": len(gt),
    })


def context_precision(sample: EvalSample) -> MetricResult:
    """C3 Context Precision (rank-aware): Average Precision over evidence list.

    AP = (1/R) * Σ_{k: relevant} (# relevant in top-k) / k
    R = |ground_truth_ids|.  Higher when relevant items appear earlier.
    """
    if not sample.ground_truth_ids:
        return MetricResult("context_precision", None, {"reason": "no_gt_ids"})

    gt = set(sample.ground_truth_ids)
    R = len(gt)

    cumulative_relevant = 0
    precision_sum = 0.0
    per_rank = []

    for k, eid in enumerate(sample.evidence_ids, start=1):
        if eid in gt:
            cumulative_relevant += 1
            prec_at_k = cumulative_relevant / k
            precision_sum += prec_at_k
            per_rank.append({"rank": k, "id": eid, "prec_at_k": round(prec_at_k, 4)})

    ap = precision_sum / R if R > 0 else 0.0

    return MetricResult("context_precision", round(ap, 4), {
        "average_precision": round(ap, 4),
        "n_relevant": R,
        "per_rank": per_rank,
    })


# ===========================================================================
# RAG answer-tier metrics (LLM-judge)
# ===========================================================================

_FAITHFULNESS_EXTRACT_PROMPT = """\
Given the following answer to a question, extract every distinct factual claim as a JSON list of strings.
Each claim should be a single, self-contained statement that can be verified independently.

Question: {question}
Answer: {answer}

Return ONLY a JSON array of strings, e.g. ["claim 1", "claim 2"].
"""

_FAITHFULNESS_VERIFY_PROMPT = """\
For each claim below, determine if it is SUPPORTED or NOT_SUPPORTED by the provided evidence.
A claim is SUPPORTED only if the evidence contains enough information to verify it.

Evidence:
{evidence}

Claims:
{claims}

Return a JSON array of objects: [{{"claim": "...", "verdict": "SUPPORTED"|"NOT_SUPPORTED"}}]
"""


async def faithfulness(sample: EvalSample) -> MetricResult:
    """C3 Faithfulness: |C⁺| / |C| — fraction of claims supported by evidence.

    Two LLM calls: (1) extract claims from answer, (2) verify each against evidence.
    """
    if not sample.answer.strip():
        return MetricResult("faithfulness", None, {"reason": "empty_answer"})

    from .llm import get_chat, LLMUnavailable
    try:
        chat = get_chat("text", temperature=0)
    except LLMUnavailable:
        return MetricResult("faithfulness", None, {"error": "llm_unavailable"})

    from langchain_core.messages import HumanMessage

    extract_msg = _FAITHFULNESS_EXTRACT_PROMPT.format(
        question=sample.question, answer=sample.answer,
    )
    resp = await chat.ainvoke([HumanMessage(content=extract_msg)])
    claims = _parse_json_list(resp.content)

    if not claims:
        return MetricResult("faithfulness", 1.0, {"reason": "no_claims_extracted"})

    evidence_block = "\n\n".join(
        f"[Evidence {i+1}]: {t}" for i, t in enumerate(sample.evidence_texts)
    ) or "(no evidence text available)"

    claims_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
    verify_msg = _FAITHFULNESS_VERIFY_PROMPT.format(
        evidence=evidence_block, claims=claims_block,
    )
    resp2 = await chat.ainvoke([HumanMessage(content=verify_msg)])
    verdicts = _parse_json_list(resp2.content, expect_objects=True)

    supported = sum(1 for v in verdicts if v.get("verdict") == "SUPPORTED")
    score = supported / len(claims)

    return MetricResult("faithfulness", round(score, 4), {
        "n_claims": len(claims),
        "supported": supported,
        "claims": claims,
        "verdicts": verdicts,
    })


# ---------------------------------------------------------------------------
# Answer Relevance — LLM-judged proxy
#
# C3 specifies: generate n reverse-questions from the answer, then compute
# mean cosine(q, q̂_i).  This proxy uses a direct LLM 0-1 rating instead.
# Label as "answer_relevance_proxy" in reports to avoid confusion with the
# C3 cosine formula.  The full C3 implementation requires an encoder and
# will be added when the evaluation encoder is available at eval time.
# ---------------------------------------------------------------------------

_ANSWER_RELEVANCE_PROMPT = """\
Rate how well the answer addresses the question on a scale of 0.0 to 1.0.

- 1.0: The answer directly and completely addresses the question.
- 0.5: The answer partially addresses the question but misses key aspects.
- 0.0: The answer is completely irrelevant or does not address the question at all.

Question: {question}
Answer: {answer}

Return ONLY a JSON object: {{"score": <float>, "reason": "<one sentence>"}}
"""


async def answer_relevance(sample: EvalSample) -> MetricResult:
    """LLM-judged answer relevance (proxy for C3 cosine formula).

    Returns score=None when LLM is unavailable.
    """
    if not sample.answer.strip():
        return MetricResult("answer_relevance_proxy", 0.0, {"reason": "empty_answer"})

    from .llm import get_chat, LLMUnavailable
    try:
        chat = get_chat("text", temperature=0)
    except LLMUnavailable:
        return MetricResult("answer_relevance_proxy", None, {"error": "llm_unavailable"})

    from langchain_core.messages import HumanMessage

    msg = _ANSWER_RELEVANCE_PROMPT.format(
        question=sample.question, answer=sample.answer,
    )
    resp = await chat.ainvoke([HumanMessage(content=msg)])
    parsed = _parse_json_object(resp.content)
    score = float(parsed.get("score", 0.0))
    score = max(0.0, min(1.0, score))

    return MetricResult("answer_relevance_proxy", round(score, 4), {
        "reason": parsed.get("reason", ""),
        "note": "LLM proxy — not C3 cosine formula",
    })


# ===========================================================================
# Supplementary: Answer Token F1 (offline)
# ===========================================================================

def answer_token_f1(sample: EvalSample) -> MetricResult:
    """Token-level F1 between predicted answer and ground truth.

    Case-insensitive. Useful baseline for short-answer VQA.
    """
    if not sample.answer.strip() or not sample.ground_truth.strip():
        return MetricResult("answer_token_f1", None, {"reason": "empty"})

    pred_tokens = _normalize_tokens(sample.answer)
    gt_tokens = _normalize_tokens(sample.ground_truth)

    common = pred_tokens & gt_tokens
    if not common:
        return MetricResult("answer_token_f1", 0.0, {
            "precision": 0.0, "recall": 0.0,
        })

    precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common) / len(gt_tokens) if gt_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return MetricResult("answer_token_f1", round(f1, 4), {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    })


# ===========================================================================
# Batch evaluation
# ===========================================================================

def evaluate_retrieval(
    sample: RetrievalSample,
    ks: list[int] | None = None,
) -> list[MetricResult]:
    """Run retrieval-tier metrics at specified K values."""
    if ks is None:
        ks = [10, 50, 100]

    results: list[MetricResult] = []
    for k in ks:
        results.append(recall_at_k(sample, k))
        results.append(precision_at_k(sample, k))
        results.append(ndcg_at_k(sample, k))
    results.append(mrr(sample))
    return results


async def evaluate_sample(
    sample: EvalSample,
    *,
    use_judge: bool = True,
) -> list[MetricResult]:
    """Run RAG context + answer metrics on one sample.

    Args:
        sample: The evaluation sample.
        use_judge: If False, skip LLM-judge metrics (faithfulness, answer_relevance).
    """
    results = [
        context_recall(sample),
        context_precision(sample),
        answer_token_f1(sample),
    ]

    if use_judge:
        results.append(await faithfulness(sample))
        results.append(await answer_relevance(sample))

    return results


def aggregate(results_per_sample: list[list[MetricResult]]) -> dict[str, float | None]:
    """Average each metric across samples, skipping None (unavailable)."""
    from collections import defaultdict
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    all_names: set[str] = set()
    for sample_results in results_per_sample:
        for r in sample_results:
            all_names.add(r.name)
            if r.score is not None:
                sums[r.name] += r.score
                counts[r.name] += 1

    return {
        name: round(sums[name] / counts[name], 4) if counts[name] > 0 else None
        for name in sorted(all_names)
    }


# ===========================================================================
# Helpers
# ===========================================================================

def _normalize_tokens(text: str) -> set[str]:
    """Lowercase, split on non-alphanumeric, drop empty."""
    return {t for t in re.split(r"\W+", text.lower()) if t}


def _parse_json_list(text: str, *, expect_objects: bool = False) -> list:
    """Extract a JSON array from LLM output, stripping markdown fences."""
    text = _strip_fences(text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return []


def _parse_json_object(text: str) -> dict:
    """Extract a JSON object from LLM output."""
    text = _strip_fences(text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


def _strip_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()

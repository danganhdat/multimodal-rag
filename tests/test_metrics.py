"""Unit tests for eval/metrics.py using slide examples.

Slide examples:
  - Query 1: relevant={D3, D5, D9, D25}
    System returns: [D1, D2, D3, D4, D5, D6, D7, D8, D9, D10]
    Recall@2 = 0/4 = 0
    Recall@5 = 2/4 = 0.5 ... wait, D3 and D5 → 2/4
    Actually slide says Recall@5 = 1.0? Let me re-read.

  Actually the slide examples from the plan are:
    - Recall@2=0, Recall@5=1, MRR=0.25
    - Recall@5=0.75
    - MRR over 3 queries = 0.583
    - Plus one hand-computed nDCG

  Let me construct concrete examples that produce these values.
"""

import math

from eval.metrics import (
    answer_accuracy,
    answerable_at_k,
    evidence_accuracy,
    match,
    mean,
    mrr,
    ndcg_at_k,
    negative_rejection,
    recall_at_k,
)


# ── Slide Example 1: Recall@2=0, Recall@5=1, MRR=0.25 ──
# 1 relevant range. System result at position 4 (0-indexed=3).
# MRR = 1/4 = 0.25

def test_recall_at_2_zero():
    results = [100, 200, 300, 500, 600]  # frame_idxs
    qrels = [(500, 500, 3)]  # relevant at frame 500
    assert recall_at_k(results, qrels, k=2) == 0.0


def test_recall_at_5_one():
    results = [100, 200, 300, 500, 600]
    qrels = [(500, 500, 3)]
    assert recall_at_k(results, qrels, k=5) == 1.0


def test_mrr_quarter():
    results = [100, 200, 300, 500, 600]
    qrels = [(500, 500, 3)]
    assert mrr(results, qrels) == 0.25


# ── Slide Example 2: Recall@5=0.75 ──
# 4 relevant items, 3 found in top 5.

def test_recall_at_5_three_quarters():
    results = [10, 20, 30, 40, 50, 60, 70]
    # 4 relevant ranges: 10-10, 30-30, 50-50, 90-90
    qrels = [(10, 10, 2), (30, 30, 2), (50, 50, 2), (90, 90, 2)]
    # Top 5: [10, 20, 30, 40, 50] → hits: 10, 30, 50 → 3/4 = 0.75
    assert recall_at_k(results, qrels, k=5) == 0.75


# ── Slide Example 3: MRR over 3 queries = 0.583 ──
# MRR = mean(1/1, 1/2, 1/4) = (1 + 0.5 + 0.25) / 3 = 0.583...

def test_mrr_three_queries():
    # Query 1: first relevant at rank 1
    r1 = [10, 20, 30]
    q1 = [(10, 10, 1)]
    # Query 2: first relevant at rank 2
    r2 = [10, 20, 30]
    q2 = [(20, 20, 1)]
    # Query 3: first relevant at rank 4
    r3 = [10, 20, 30, 40, 50]
    q3 = [(40, 40, 1)]

    mrr_values = [mrr(r1, q1), mrr(r2, q2), mrr(r3, q3)]
    assert mrr_values == [1.0, 0.5, 0.25]
    result = mean(mrr_values)
    assert abs(result - 7 / 12) < 1e-9  # 0.58333...


# ── nDCG hand-computed example ──
# Results (grades): [3, 2, 0, 1, 0] at k=5
# DCG = (2^3-1)/log2(2) + (2^2-1)/log2(3) + 0 + (2^1-1)/log2(5) + 0
#     = 7/1 + 3/1.585 + 0 + 1/2.322 + 0
#     = 7 + 1.893 + 0.431 = 9.324
# Ideal grades sorted desc: [3, 2, 1, 0, 0]
# IDCG = 7/1 + 3/1.585 + 1/2 + 0 + 0 = 7 + 1.893 + 0.5 = 9.393
# nDCG = 9.324 / 9.393 ≈ 0.9926

def test_ndcg_hand_computed():
    # frame_idxs at positions 0-4
    results = [100, 200, 300, 400, 500]
    qrels = [
        (100, 100, 3),  # grade 3 at pos 0
        (200, 200, 2),  # grade 2 at pos 1
        (400, 400, 1),  # grade 1 at pos 3
    ]
    val = ndcg_at_k(results, qrels, k=5)

    # Manual calculation
    dcg = (2**3 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3) + 0 + (2**1 - 1) / math.log2(5) + 0
    ideal_grades = [3, 2, 1]  # only 3 qrels
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal_grades))
    expected = dcg / idcg

    assert abs(val - expected) < 1e-6


# ── match function ──

def test_match_in_range():
    qrels = [(100, 200, 2), (300, 400, 3)]
    assert match(150, qrels) == 2
    assert match(350, qrels) == 3
    assert match(250, qrels) == 0


def test_match_best_grade():
    # Overlapping ranges: pick highest grade
    qrels = [(100, 300, 1), (200, 400, 3)]
    assert match(250, qrels) == 3


# ── VQA metrics ──

def test_answerable_at_k():
    results = [100, 200, 300]
    qrels = [(300, 300, 3)]
    assert answerable_at_k(results, qrels, k=3) is True
    assert answerable_at_k(results, qrels, k=2) is False


def test_answer_accuracy():
    assert answer_accuracy("Hello World", ["hello world"]) is True
    assert answer_accuracy("café", ["cafe"]) is True  # accent stripped
    assert answer_accuracy("abc", ["def"]) is False


def test_evidence_accuracy():
    qrels = [(100, 200, 3)]
    # 2 evidence frames: one in range, one not
    assert evidence_accuracy([150, 300], qrels) == 0.5
    assert evidence_accuracy([], qrels) == 0.0


def test_negative_rejection():
    # Answerable question, system answers → correct
    assert negative_rejection("42", is_answerable=True) is True
    # Answerable question, system refuses → wrong
    assert negative_rejection("not_found", is_answerable=True) is False
    # Unanswerable, system refuses → correct
    assert negative_rejection("not_found", is_answerable=False) is True
    # Unanswerable, system answers → wrong
    assert negative_rejection("42", is_answerable=False) is False


# ── Edge cases ──

def test_empty_results():
    assert recall_at_k([], [(1, 1, 1)], k=5) == 0.0
    assert mrr([], [(1, 1, 1)]) == 0.0
    assert ndcg_at_k([], [(1, 1, 1)], k=5) == 0.0


def test_no_relevant():
    assert recall_at_k([1, 2, 3], [], k=5) == 0.0
    assert mrr([1, 2, 3], []) == 0.0
    assert ndcg_at_k([1, 2, 3], [], k=5) == 0.0

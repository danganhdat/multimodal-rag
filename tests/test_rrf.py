"""Unit tests for client-side RRF fusion logic."""


def rrf_score(ranks: list[int], k: int = 60) -> float:
    """Compute RRF score from a list of ranks (0-indexed)."""
    return sum(1.0 / (k + r + 1) for r in ranks)


def rrf_merge(lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Merge multiple ranked lists using RRF.

    Each list is a sequence of item IDs ordered by rank.
    Returns [(id, score)] sorted by score descending.
    """
    from collections import defaultdict

    rank_map: defaultdict[str, list[int]] = defaultdict(list)
    for lst in lists:
        for rank, item_id in enumerate(lst):
            rank_map[item_id].append(rank)

    scored = [(item_id, rrf_score(ranks, k)) for item_id, ranks in rank_map.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def test_single_list():
    result = rrf_merge([["A", "B", "C"]], k=60)
    assert result[0][0] == "A"
    assert result[1][0] == "B"
    assert result[2][0] == "C"
    # Scores should be decreasing
    assert result[0][1] > result[1][1] > result[2][1]


def test_two_lists_agreement():
    """Items in both lists should rank higher than items in only one."""
    result = rrf_merge([["A", "B", "C"], ["A", "C", "D"]], k=60)
    scores = {r[0]: r[1] for r in result}
    # A is rank 0 in both → highest
    assert scores["A"] > scores["B"]
    assert scores["A"] > scores["D"]
    # C is in both but at different positions
    assert scores["C"] > scores["D"]  # C in both, D in only one


def test_rrf_score_value():
    # rank 0 in one list, rank 2 in another, k=60
    score = rrf_score([0, 2], k=60)
    expected = 1 / 61 + 1 / 63
    assert abs(score - expected) < 1e-10


def test_disjoint_lists():
    """Completely disjoint lists → each item has score from one list only."""
    result = rrf_merge([["A", "B"], ["C", "D"]], k=60)
    scores = {r[0]: r[1] for r in result}
    # A at rank 0 in list 1, C at rank 0 in list 2 → same score
    assert abs(scores["A"] - scores["C"]) < 1e-10
    assert abs(scores["B"] - scores["D"]) < 1e-10


def test_empty_list():
    result = rrf_merge([], k=60)
    assert result == []


def test_symmetric_rrf():
    """Symmetric input lists → symmetric scores for mirrored items."""
    result = rrf_merge([["A", "B", "C"], ["C", "B", "A"]], k=60)
    scores = {r[0]: r[1] for r in result}
    # A: ranks (0,2), C: ranks (2,0) → same score (symmetric)
    assert abs(scores["A"] - scores["C"]) < 1e-10
    # B: ranks (1,1) → score = 2/(k+2)
    # A: ranks (0,2) → score = 1/(k+1) + 1/(k+3)
    # With k=60: B = 2/62 ≈ 0.03226, A = 1/61 + 1/63 ≈ 0.03227
    # A slightly wins because 1/61 contributes more than the loss of 1/63 vs 1/62
    assert scores["A"] >= scores["B"]

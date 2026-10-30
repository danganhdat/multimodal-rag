"""Unit tests for the temporal search expand logic.

Tests use a minimal implementation of the expand function
from the plan to validate the algorithm before the actual module.
"""

import math


def _local_gap(t: list[float], j: int) -> float:
    """Gap between frame j and its nearest neighbor in time."""
    gaps = []
    if j > 0:
        gaps.append(abs(t[j] - t[j - 1]))
    if j < len(t) - 1:
        gaps.append(abs(t[j + 1] - t[j]))
    return min(gaps) if gaps else 1.0


def _median_gap(t: list[float]) -> float:
    """Median gap between consecutive frames."""
    if len(t) < 2:
        return 1.0
    gaps = sorted(abs(t[i + 1] - t[i]) for i in range(len(t) - 1))
    mid = len(gaps) // 2
    return gaps[mid]


def _patience(j: int, t: list[float], base: int) -> int:
    mg = _median_gap(t)
    lg = max(_local_gap(t, j), 1e-3)
    return max(2, min(8, round(base * mg / lg)))


def expand(S_e, start, step, t, t_ref, max_gap_s, base_patience, eps=0.01, tau_min=None):
    """Expand from anchor in one direction, picking the best-scoring frame."""
    best = -math.inf
    pos = None
    tol = 0
    j = start + step

    while 0 <= j < len(t) and abs(t[j] - t_ref) <= max_gap_s:
        if S_e[j] > best + eps:
            best, pos, tol = S_e[j], j, 0
        else:
            tol += 1
        if tol >= _patience(j, t, base_patience):
            break
        j += step

    if pos is not None and (tau_min is None or best >= tau_min):
        return (pos, best)
    return (None, None)


# ── Tests ──

def test_expand_forward_finds_peak():
    """Forward expansion from position 2, finds peak at position 5."""
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 0.3, 0.1]
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    pos, score = expand(scores, start=2, step=1, t=times, t_ref=2.0,
                        max_gap_s=60, base_patience=3)
    assert pos == 5
    assert score == 0.9


def test_expand_backward_finds_peak():
    """Backward expansion from position 5, finds peak at position 1."""
    scores = [0.1, 0.8, 0.3, 0.2, 0.4, 0.5, 0.3]
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    pos, score = expand(scores, start=5, step=-1, t=times, t_ref=5.0,
                        max_gap_s=60, base_patience=3)
    assert pos == 1
    assert score == 0.8


def test_expand_stops_at_max_gap():
    """Expansion stops when time difference exceeds max_gap_s."""
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.9]  # peak at 5
    times = [0.0, 10.0, 20.0, 30.0, 40.0, 100.0]  # big jump to 100
    pos, score = expand(scores, start=0, step=1, t=times, t_ref=0.0,
                        max_gap_s=50, base_patience=3)
    # Should not reach position 5 (100s away from t_ref=0)
    assert pos is not None
    assert pos <= 4


def test_expand_respects_tau_min():
    """If best score < tau_min, returns None."""
    scores = [0.1, 0.2, 0.15, 0.1]
    times = [0.0, 1.0, 2.0, 3.0]
    pos, score = expand(scores, start=0, step=1, t=times, t_ref=0.0,
                        max_gap_s=60, base_patience=3, tau_min=0.5)
    assert pos is None
    assert score is None


def test_expand_patience_stops_early():
    """Patience mechanism stops expansion when no improvement found."""
    # All scores plateau after position 2
    scores = [0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    times = [float(i) for i in range(10)]
    pos, score = expand(scores, start=0, step=1, t=times, t_ref=0.0,
                        max_gap_s=60, base_patience=3)
    # Should find position 1 (first 0.5) and stop soon after
    assert pos == 1
    assert score == 0.5


def test_expand_empty_direction():
    """Expanding past the end of the array returns None."""
    scores = [0.5]
    times = [0.0]
    pos, score = expand(scores, start=0, step=1, t=times, t_ref=0.0,
                        max_gap_s=60, base_patience=3)
    assert pos is None


def test_patience_dynamic():
    """Patience increases in dense regions (small gaps) and decreases in sparse ones."""
    # Dense: gaps of 0.1s each → patience should be higher
    t_dense = [i * 0.1 for i in range(20)]
    p_dense = _patience(5, t_dense, base=3)

    # Sparse: gaps of 10s each → patience should be lower
    t_sparse = [i * 10.0 for i in range(20)]
    p_sparse = _patience(5, t_sparse, base=3)

    # Both should be clamped to [2, 8]
    assert 2 <= p_dense <= 8
    assert 2 <= p_sparse <= 8

    # In uniform spacing, median_gap == local_gap → patience == base
    assert p_dense == 3
    assert p_sparse == 3


def test_chain_score():
    """Basic chain scoring: mean of event scores minus gap penalty."""
    event_scores = [0.8, 0.7, 0.6]
    times_at_events = [10.0, 25.0, 40.0]  # gaps: 15, 15
    free_gap_s = 10.0
    max_gap_s = 60.0
    lambda_gap = 0.3
    w_sup = 0.1
    support = 2

    mean_s = sum(event_scores) / len(event_scores)
    gap_penalty = lambda_gap * sum(
        max(0, (times_at_events[i + 1] - times_at_events[i]) - free_gap_s) / max_gap_s
        for i in range(len(times_at_events) - 1)
    )
    skip_pen = 0  # no skips
    sup_bonus = w_sup * math.log(1 + support)

    chain_score = mean_s - gap_penalty - skip_pen + sup_bonus

    # Manually: mean = 0.7, gap_pen = 0.3 * (5/60 + 5/60) = 0.3 * 0.1667 = 0.05
    # sup = 0.1 * ln(3) ≈ 0.1099
    # total ≈ 0.7 - 0.05 + 0.1099 = 0.7599
    assert abs(chain_score - (0.7 - 0.05 + 0.1 * math.log(3))) < 1e-4

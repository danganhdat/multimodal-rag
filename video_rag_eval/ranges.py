"""Project verified keyframe qrels to tolerated original-video frame intervals."""

from __future__ import annotations

from collections import defaultdict


def interval(frame_idx: int, ordered_frame_idxs: list[int], video_last_frame: int | None = None) -> tuple[int, int]:
    """Adjacent keyframes are excluded; both returned bounds are inclusive."""
    xs = sorted(set(ordered_frame_idxs))
    if frame_idx not in xs or not xs:
        raise ValueError("frame_idx missing in video keyframe map")
    i = xs.index(frame_idx)
    lo = xs[i - 1] + 1 if i else xs[0]
    hi = xs[i + 1] - 1 if i + 1 < len(xs) else (video_last_frame if video_last_frame is not None else xs[-1])
    if lo > frame_idx or hi < frame_idx:
        raise ValueError("invalid keyframe neighborhood")
    return lo, hi


def make_ranges(qrels: list[dict], maps: dict[str, list[int]], video_ends: dict[str, int] | None = None) -> list[dict]:
    result = []
    video_ends = video_ends or {}
    for q in qrels:
        fid = q["frame_id"]
        video, n = fid.split("/")
        xs = maps.get(video)
        if not xs or int(n) < 1 or int(n) > len(xs):
            raise ValueError(f"missing map for {fid}")
        fi = xs[int(n) - 1]
        lo, hi = interval(fi, xs, video_ends.get(video))
        result.append({"query_id": q["query_id"], "video_id": video,
                       "lo_frame_idx": lo, "hi_frame_idx": hi,
                       "grade": int(q["grade"]), "source_frame_id": fid})
    return result


def _disjoint(ranges: list[dict]) -> list[dict]:
    """Resolve overlaps by maximum grade into nonoverlapping, graded segments."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in ranges:
        groups[(r["query_id"], r["video_id"])].append(r)
    result = []
    for (qid, vid), rows in groups.items():
        boundaries = sorted({x for r in rows for x in (int(r["lo_frame_idx"]), int(r["hi_frame_idx"]) + 1)})
        for lo, end in zip(boundaries, boundaries[1:]):
            covering = [r for r in rows if r["lo_frame_idx"] <= lo <= r["hi_frame_idx"]]
            if not covering:
                continue
            grade = max(r["grade"] for r in covering)
            if result and result[-1]["query_id"] == qid and result[-1]["video_id"] == vid and result[-1]["grade"] == grade and result[-1]["hi_frame_idx"] + 1 == lo:
                result[-1]["hi_frame_idx"] = end - 1
            else:
                result.append({"query_id": qid, "video_id": vid, "lo_frame_idx": lo,
                               "hi_frame_idx": end - 1, "grade": grade})
    return result


def match(query_id: str, video_id: str, frame_idx: int, ranges: list[dict]) -> int | None:
    """None means unjudged, while zero means explicitly judged irrelevant."""
    grades = [r["grade"] for r in ranges if r["query_id"] == query_id and
              r["video_id"] == video_id and r["lo_frame_idx"] <= frame_idx <= r["hi_frame_idx"]]
    return max(grades) if grades else None


def positive_units(query_id: str, ranges: list[dict], threshold: int = 2) -> list[dict]:
    return [r for r in _disjoint(ranges) if r["query_id"] == query_id and r["grade"] >= threshold]

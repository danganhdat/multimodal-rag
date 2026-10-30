"""Keyframe map utilities: load CSV mappings, convert between n and frame_idx."""

import csv
from pathlib import Path


def load_keyframe_map(map_dir: Path) -> dict[str, dict]:
    """Load all map-keyframe CSVs.

    Returns: {video_name: {n: {frame_idx, pts_time, fps}}}
    """
    result: dict[str, dict] = {}
    if not map_dir.exists():
        return result
    for csv_path in sorted(map_dir.glob("*.csv")):
        video = csv_path.stem
        video_map = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                n = int(r["n"])
                video_map[n] = {
                    "frame_idx": int(r["frame_idx"]),
                    "pts_time": float(r["pts_time"]),
                    "fps": float(r["fps"]),
                }
        result[video] = video_map
    return result


def kf_id_to_frame_idx(kf_id: str, kf_map: dict[str, dict]) -> int | None:
    """Convert a kf_id like 'L21_V001/004' to its frame_idx using the CSV map."""
    parts = kf_id.split("/")
    if len(parts) != 2:
        return None
    video = parts[0]
    try:
        n = int(parts[1])
    except ValueError:
        return None
    vm = kf_map.get(video, {})
    entry = vm.get(n)
    return entry["frame_idx"] if entry else None


def get_neighbor_kf_ids(
    kf_id: str,
    kf_map: dict[str, dict],
    direction: str = "both",
    n_neighbors: int = 1,
) -> list[str]:
    """Get neighbor keyframe IDs from the CSV map.

    direction: "forward", "backward", or "both".
    Returns kf_ids sorted by n.
    """
    parts = kf_id.split("/")
    if len(parts) != 2:
        return []
    video = parts[0]
    try:
        n = int(parts[1])
    except ValueError:
        return []

    vm = kf_map.get(video, {})
    if not vm:
        return []

    all_ns = sorted(vm.keys())
    try:
        idx = all_ns.index(n)
    except ValueError:
        return []

    neighbors = []
    if direction in ("backward", "both"):
        for i in range(max(0, idx - n_neighbors), idx):
            neighbors.append(f"{video}/{all_ns[i]:03d}")
    if direction in ("forward", "both"):
        for i in range(idx + 1, min(len(all_ns), idx + 1 + n_neighbors)):
            neighbors.append(f"{video}/{all_ns[i]:03d}")

    return neighbors

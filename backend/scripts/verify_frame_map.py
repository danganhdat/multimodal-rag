"""
Read-only verification: compare frame_idx / pts_time in Milvus
against map-keyframes CSVs.

Reports:
  - Videos in Milvus with no CSV file
  - Keyframes where Milvus frame_idx != CSV frame_idx
  - Keyframes where n > 1 and pts_time == 0 (likely fallback data)
  - CSVs with no matching Milvus data

Usage:
    python -m backend.scripts.verify_frame_map
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

from pymilvus import MilvusClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings


def load_csv_map(map_dir: Path) -> dict[str, dict]:
    """Load all map-keyframe CSVs → {video/nnn: {frame_idx, pts_time, fps}}."""
    result = {}
    if not map_dir.exists():
        return result
    for csv_path in sorted(map_dir.glob("*.csv")):
        video = csv_path.stem
        with open(csv_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                n = int(r["n"])
                key = f"{video}/{n:03d}"
                result[key] = {
                    "frame_idx": int(r["frame_idx"]),
                    "pts_time": float(r["pts_time"]),
                    "fps": float(r["fps"]),
                }
    return result


def iter_milvus(client: MilvusClient, collection: str, batch: int = 5000):
    """Iterate all rows in Milvus collection using query_iterator."""
    it = client.query_iterator(
        collection_name=collection,
        output_fields=["id", "video_name", "keyframe_idx", "frame_idx", "pts_time"],
        batch_size=batch,
    )
    while True:
        rows = it.next()
        if not rows:
            break
        yield from rows
    it.close()


def main():
    map_dir = settings.MAP_KEYFRAMES_DIR
    print(f"MAP_KEYFRAMES_DIR: {map_dir}")
    print(f"MILVUS_URI:        {settings.MILVUS_URI}")
    print(f"COLLECTION:        {settings.COLLECTION_NAME}")
    print()

    csv_map = load_csv_map(map_dir)
    print(f"CSV entries loaded: {len(csv_map)}")

    if not csv_map:
        print("WARNING: No CSV files found. Cannot verify.")
        return

    csv_videos = {k.split("/")[0] for k in csv_map}
    print(f"CSV videos: {len(csv_videos)}")
    print()

    client = MilvusClient(uri=settings.MILVUS_URI)

    mismatch_frame_idx = []
    pts_zero = []
    no_csv = defaultdict(int)
    milvus_videos = set()
    total = 0

    print("Scanning Milvus rows...")
    for row in iter_milvus(client, settings.COLLECTION_NAME):
        total += 1
        rid = row["id"]
        video = row["video_name"]
        n = row["keyframe_idx"]
        milvus_videos.add(video)

        csv_entry = csv_map.get(rid)
        if csv_entry is None:
            no_csv[video] += 1
            continue

        # Check frame_idx match
        if row["frame_idx"] != csv_entry["frame_idx"]:
            mismatch_frame_idx.append({
                "id": rid,
                "milvus_frame_idx": row["frame_idx"],
                "csv_frame_idx": csv_entry["frame_idx"],
            })

        # Check suspicious pts_time = 0 for n > 1
        if n > 1 and row["pts_time"] == 0.0:
            pts_zero.append(rid)

    print(f"\nMilvus rows scanned: {total}")
    print(f"Milvus videos: {len(milvus_videos)}")
    print()

    # --- Report ---
    # Videos in Milvus without CSV
    videos_no_csv = milvus_videos - csv_videos
    if videos_no_csv:
        print(f"WARN: {len(videos_no_csv)} videos in Milvus have NO CSV file:")
        for v in sorted(videos_no_csv)[:20]:
            print(f"  {v} ({no_csv[v]} keyframes)")
        if len(videos_no_csv) > 20:
            print(f"  ... and {len(videos_no_csv) - 20} more")
    else:
        print("OK: All Milvus videos have a CSV file.")

    # CSVs without Milvus data
    csv_only = csv_videos - milvus_videos
    if csv_only:
        print(f"\nINFO: {len(csv_only)} CSV videos not in Milvus (expected if not ingested):")
        for v in sorted(csv_only)[:10]:
            print(f"  {v}")
        if len(csv_only) > 10:
            print(f"  ... and {len(csv_only) - 10} more")

    # frame_idx mismatches
    if mismatch_frame_idx:
        print(f"\nERROR: {len(mismatch_frame_idx)} keyframes with frame_idx mismatch:")
        for m in mismatch_frame_idx[:20]:
            print(f"  {m['id']}: milvus={m['milvus_frame_idx']} csv={m['csv_frame_idx']}")
        if len(mismatch_frame_idx) > 20:
            print(f"  ... and {len(mismatch_frame_idx) - 20} more")
    else:
        print("\nOK: All frame_idx values match CSV.")

    # pts_time == 0 for n > 1
    if pts_zero:
        print(f"\nWARN: {len(pts_zero)} keyframes with n>1 and pts_time==0 (likely fallback):")
        for p in pts_zero[:20]:
            print(f"  {p}")
        if len(pts_zero) > 20:
            print(f"  ... and {len(pts_zero) - 20} more")
    else:
        print("OK: No suspicious pts_time=0 for n>1.")

    # Keyframes in Milvus that have no CSV entry at all
    total_no_csv = sum(no_csv.values())
    if total_no_csv:
        print(f"\nWARN: {total_no_csv} keyframes in Milvus have no CSV entry.")
    else:
        print("OK: All Milvus keyframes have a CSV entry.")

    # Summary
    print("\n" + "=" * 50)
    issues = len(mismatch_frame_idx) + len(pts_zero) + len(videos_no_csv) + total_no_csv
    if issues == 0:
        print("PASS: No issues found.")
    else:
        print(f"DONE: {issues} issue(s) found. Review above.")


if __name__ == "__main__":
    main()

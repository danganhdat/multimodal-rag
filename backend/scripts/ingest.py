"""
Ingest features from pkl into Milvus.

Uses sketch_dense (512-dim, TSBIR ViT-B-16) and text_dense (768-dim, SigLIP2)
vectors, plus OCR and objects metadata.

Usage:
    python -m backend.scripts.ingest
"""

import csv
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pymilvus import DataType, MilvusClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings

CHUNK_SIZE = 1000
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PKL_PATH = PROJECT_ROOT / "clip-features-32" / "aic25_batch1_features.pkl"


def create_collection(client, name):
    if client.has_collection(name):
        client.drop_collection(name)
        print(f"Dropped: {name}")

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=32)
    schema.add_field("video_name", DataType.VARCHAR, max_length=16)
    schema.add_field("keyframe_idx", DataType.INT32)
    schema.add_field("frame_idx", DataType.INT64)
    schema.add_field("pts_time", DataType.FLOAT)
    schema.add_field("fps", DataType.FLOAT)
    schema.add_field("clip_vector", DataType.FLOAT_VECTOR, dim=512)
    schema.add_field("text_vector", DataType.FLOAT_VECTOR, dim=768)
    schema.add_field("objects", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=20, max_length=64)
    schema.add_field("ocr", DataType.VARCHAR, max_length=4096, default_value="")

    metric = os.environ.get("MILVUS_METRIC", "COSINE")
    idx = client.prepare_index_params()
    idx.add_index(
        field_name="clip_vector",
        index_type="HNSW",
        metric_type=metric,
        params={"M": 32, "efConstruction": 256},
    )
    idx.add_index(
        field_name="text_vector",
        index_type="HNSW",
        metric_type=metric,
        params={"M": 32, "efConstruction": 256},
    )

    client.create_collection(collection_name=name, schema=schema, index_params=idx)
    print(f"Created: {name}")


def load_csv_map(map_dir):
    """Load all map-keyframe CSVs into a dict: {video/kf -> {frame_idx, pts_time, fps}}."""
    result = {}
    if not map_dir.exists():
        return result
    for csv_path in sorted(map_dir.glob("*.csv")):
        video = csv_path.stem
        with open(csv_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = f"{video}/{int(r['n']):03d}"
                result[key] = {
                    "frame_idx": int(r["frame_idx"]),
                    "pts_time": float(r["pts_time"]),
                    "fps": float(r["fps"]),
                }
    return result


def is_blank_frame(img_path: Path) -> bool:
    try:
        if not img_path.exists():
            return False
        if img_path.stat().st_size > 60_000:
            return False
        img = Image.open(img_path)
        img.draft("RGB", (16, 16))
        arr = np.array(img.convert("RGB")).astype(np.float32)
        mean, std = arr.mean(), arr.std()
        return (mean > 220 and std < 30) or (mean < 15 and std < 10)
    except Exception:
        return False


def ingest():
    client = MilvusClient(uri=settings.MILVUS_URI)
    create_collection(client, settings.COLLECTION_NAME)

    csv_map = load_csv_map(settings.MAP_KEYFRAMES_DIR)
    if csv_map:
        print(f"Map keyframes: {len(csv_map)} entries")
    else:
        print("Map keyframes not available, using defaults for frame_idx/pts_time/fps")

    print(f"Loading features from {PKL_PATH}...")
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"PKL entries: {len(data)}")

    chunk, inserted, skipped, objects_set = [], 0, 0, set()

    for i, item in enumerate(data):
        m = re.match(r"(L\d+_V\d+)/(\d+)\.jpg$", item["path"])
        if not m:
            continue

        video = m.group(1)
        kf = int(m.group(2))
        key = f"{video}/{kf:03d}"

        img_path = settings.KEYFRAMES_DIR / video / f"{kf:03d}.jpg"
        if is_blank_frame(img_path):
            skipped += 1
            continue

        meta = csv_map.get(key, {"frame_idx": kf, "pts_time": 0.0, "fps": 25.0})
        ocr_text = item.get("ocr", "").replace("<sep>", " ").strip()
        objs = item.get("objects", [])[:20]
        objects_set.update(objs)

        chunk.append({
            "id": key,
            "video_name": video,
            "keyframe_idx": kf,
            "frame_idx": meta["frame_idx"],
            "pts_time": meta["pts_time"],
            "fps": meta["fps"],
            "clip_vector": item["sketch_dense"],
            "text_vector": item["text_dense"],
            "objects": objs,
            "ocr": ocr_text[:4096],
        })

        if len(chunk) >= CHUNK_SIZE:
            client.insert(settings.COLLECTION_NAME, chunk)
            inserted += len(chunk)
            chunk.clear()

        if (i + 1) % 20000 == 0:
            print(f"  [{i+1}/{len(data)}] — inserted: {inserted + len(chunk)}")

    if chunk:
        client.insert(settings.COLLECTION_NAME, chunk)
        inserted += len(chunk)

    client.flush(settings.COLLECTION_NAME)
    stats = client.get_collection_stats(settings.COLLECTION_NAME)
    print(f"\nDone. Inserted: {inserted}, Skipped blank: {skipped}, "
          f"Milvus rows: {stats['row_count']}, Objects: {len(objects_set)}")

    obj_path = PROJECT_ROOT / "object_classes.json"
    obj_path.write_text(
        json.dumps(sorted(objects_set), ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    ingest()

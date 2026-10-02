"""
Data ingestion pipeline: load CLIP features, map-keyframes CSVs,
object detections, and insert into Milvus.

Usage:
    cd backend
    python -m scripts.ingest
"""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from pymilvus import DataType, MilvusClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings


def create_collection(client: MilvusClient, name: str) -> None:
    if client.has_collection(name):
        client.drop_collection(name)
        print(f"Dropped existing collection: {name}")

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)

    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=32)
    schema.add_field("video_name", DataType.VARCHAR, max_length=16)
    schema.add_field("keyframe_idx", DataType.INT32)
    schema.add_field("frame_idx", DataType.INT64)
    schema.add_field("pts_time", DataType.FLOAT)
    schema.add_field("fps", DataType.FLOAT)
    schema.add_field("clip_vector", DataType.FLOAT_VECTOR, dim=512)
    schema.add_field(
        "objects",
        DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=20,
        max_length=64,
    )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="clip_vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=name,
        schema=schema,
        index_params=index_params,
    )
    print(f"Created collection: {name}")


def load_csv_mapping(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "n": int(row["n"]),
                    "pts_time": float(row["pts_time"]),
                    "fps": float(row["fps"]),
                    "frame_idx": int(row["frame_idx"]),
                }
            )
    return rows


def load_objects(json_path: Path, score_threshold: float, max_count: int) -> list[str]:
    if not json_path.exists():
        return []
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []

    scores = data.get("detection_scores", [])
    entities = data.get("detection_class_entities", [])
    seen = set()
    result = []
    for score_str, entity in zip(scores, entities):
        score = float(score_str)
        if score < score_threshold:
            break
        if entity not in seen:
            seen.add(entity)
            result.append(entity)
            if len(result) >= max_count:
                break
    return result


def ingest() -> None:
    client = MilvusClient(uri=settings.MILVUS_URI)
    create_collection(client, settings.COLLECTION_NAME)

    feature_files = sorted(settings.CLIP_FEATURES_DIR.glob("*.npy"))
    print(f"Found {len(feature_files)} feature files")

    all_objects: set[str] = set()
    chunk: list[dict] = []
    chunk_size = 1000
    total_inserted = 0
    total_videos = len(feature_files)

    for vi, feat_path in enumerate(feature_files):
        video_name = feat_path.stem
        features = np.load(feat_path).astype(np.float32)
        n_keyframes = features.shape[0]

        csv_path = settings.MAP_KEYFRAMES_DIR / f"{video_name}.csv"
        if not csv_path.exists():
            print(f"  SKIP {video_name}: no CSV mapping")
            continue
        csv_rows = load_csv_mapping(csv_path)

        if len(csv_rows) != n_keyframes:
            print(
                f"  WARN {video_name}: CSV rows ({len(csv_rows)}) != "
                f"features ({n_keyframes}), using min"
            )

        count = min(len(csv_rows), n_keyframes)

        for i in range(count):
            kf_idx = csv_rows[i]["n"]
            obj_path = (
                settings.OBJECTS_DIR / video_name / f"{kf_idx:03d}.json"
            )
            objects = load_objects(
                obj_path,
                settings.OBJECT_SCORE_THRESHOLD,
                settings.OBJECT_MAX_PER_FRAME,
            )
            all_objects.update(objects)

            record = {
                "id": f"{video_name}/{kf_idx:03d}",
                "video_name": video_name,
                "keyframe_idx": kf_idx,
                "frame_idx": csv_rows[i]["frame_idx"],
                "pts_time": csv_rows[i]["pts_time"],
                "fps": csv_rows[i]["fps"],
                "clip_vector": features[i].tolist(),
                "objects": objects,
            }
            chunk.append(record)

            if len(chunk) >= chunk_size:
                client.insert(settings.COLLECTION_NAME, chunk)
                total_inserted += len(chunk)
                chunk.clear()

        if (vi + 1) % 50 == 0 or vi == total_videos - 1:
            print(
                f"  [{vi + 1}/{total_videos}] {video_name} "
                f"({n_keyframes} kf) — total inserted so far: {total_inserted + len(chunk)}"
            )

    if chunk:
        client.insert(settings.COLLECTION_NAME, chunk)
        total_inserted += len(chunk)
        chunk.clear()

    client.flush(settings.COLLECTION_NAME)
    stats = client.get_collection_stats(settings.COLLECTION_NAME)
    print(f"\nDone. Total inserted: {total_inserted}")
    print(f"Milvus row count: {stats['row_count']}")
    print(f"Unique object classes: {len(all_objects)}")

    obj_path = settings.DATA_ROOT / "object_classes.json"
    obj_path.write_text(
        json.dumps(sorted(all_objects), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved object classes to {obj_path}")


if __name__ == "__main__":
    ingest()

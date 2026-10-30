"""Corpus inventory, safe visual deduplication and deterministic pilot sampling."""

from __future__ import annotations

import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.fft import dctn

from .core import output, read_jsonl, source, write_json, write_jsonl


def load_maps(map_root: Path) -> tuple[dict[str, dict[int, dict]], list[dict]]:
    maps, problems = {}, []
    for path in sorted(map_root.glob("*.csv")):
        rows = {}
        try:
            with path.open(newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    n, fi = int(row["n"]), int(row["frame_idx"])
                    if n in rows:
                        raise ValueError(f"duplicate n={n}")
                    rows[n] = {"frame_idx": fi, "pts_time": float(row["pts_time"]), "fps": float(row["fps"])}
            ns = sorted(rows)
            fis = [rows[n]["frame_idx"] for n in ns]
            if ns != list(range(1, len(ns) + 1)) or fis != sorted(fis):
                raise ValueError("n is not contiguous or frame_idx decreases")
            pts = [rows[n]["pts_time"] for n in ns]
            if pts != sorted(pts) or any(rows[n]["fps"] <= 0 for n in ns):
                raise ValueError("pts_time decreases or fps is nonpositive")
            maps[path.stem] = rows
        except (ValueError, KeyError, OSError) as e:
            problems.append({"video_id": path.stem, "reason": "invalid_map", "detail": str(e)})
    return maps, problems


def phash(image: Image.Image) -> str:
    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    freq = dctn(gray, type=2, norm="ortho")[:8, :8]
    values = freq.ravel()[1:]
    bits = values > np.median(values)
    return f"{int(''.join('1' if x else '0' for x in bits), 2):016x}"


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(cfg: dict, max_videos: int | None = None, source_only: bool = False) -> dict:
    images = source(cfg, "keyframes")
    maps, quarantine = load_maps(source(cfg, "maps"))
    if not images.is_dir():
        raise FileNotFoundError(images)
    video_dirs = sorted(p for p in images.iterdir() if p.is_dir())
    if source_only:
        video_dirs = [p for p in video_dirs if (source(cfg, "videos") / f"{p.name}.mp4").is_file()]
    if max_videos is not None:
        video_dirs = video_dirs[:max_videos]
    frames = []
    for folder in video_dirs:
        video, mapping = folder.name, maps.get(folder.name, {})
        for image_path in sorted(folder.glob("*.jpg")):
            try:
                n = int(image_path.stem)
                if n not in mapping:
                    raise ValueError("missing CSV row for image")
                with Image.open(image_path) as im:
                    im.verify()
                with Image.open(image_path) as im:
                    width, height = im.size
                    fingerprint = phash(im)
                fid = f"{video}/{n:03d}"
                info = mapping[n]
                obj = source(cfg, "objects") / video / f"{n:03d}.json"
                video_path = source(cfg, "videos") / f"{video}.mp4"
                if obj.exists():
                    try:
                        import json
                        json.loads(obj.read_text(encoding="utf-8"))
                    except (ValueError, OSError, UnicodeError) as e:
                        quarantine.append({"video_id": video, "object_path": str(obj),
                                           "reason": "invalid_object_json", "detail": str(e)})
                        obj = Path("__invalid__")
                frames.append({"frame_id": fid, "video_id": video, "n": n,
                               "frame_idx": info["frame_idx"], "pts_time": info["pts_time"],
                               "fps": info["fps"], "image_path": str(image_path),
                               "object_path": str(obj) if obj.exists() else None,
                               "metadata_available": (source(cfg, "media_info") / f"{video}.json").is_file(),
                               "video_path": str(video_path) if video_path.exists() else None,
                               "width": width, "height": height, "sha256": _digest(image_path),
                               "phash": fingerprint, "temporal_eligible": True})
            except (ValueError, OSError, UnidentifiedImageError) as e:
                quarantine.append({"video_id": video, "image_path": str(image_path),
                                   "reason": "invalid_frame", "detail": str(e)})
    write_jsonl(output(cfg, "derived", "frames.jsonl"), frames)
    write_jsonl(output(cfg, "derived", "quarantine.jsonl"), quarantine)
    try:
        import pandas as pd
        pd.DataFrame(frames).to_parquet(output(cfg, "derived", "frames.parquet"), index=False)
        parquet = True
    except ImportError:
        parquet = False
    report = {"videos_scanned": len(video_dirs), "videos_available": len([p for p in images.iterdir() if p.is_dir()]),
              "source_only": source_only,
              "frames_valid": len(frames),
              "quarantine_count": len(quarantine), "csv_videos_available": len(maps),
              "source_video_count": len({r["video_id"] for r in frames if r["video_path"]}),
              "object_coverage": sum(bool(r["object_path"]) for r in frames) / len(frames) if frames else 0,
              "metadata_coverage": sum(r["metadata_available"] for r in frames) / len(frames) if frames else 0,
              "parquet_written": parquet,
              "frame_id_format": "video_id/keyframe_n_zero_padded_3",
              "timestamp_source": "D:/map-keyframes/{video}.csv"}
    write_json(output(cfg, "derived", "dataset_report.json"), report)
    return report


def _ssim(a: Path, b: Path) -> float:
    with Image.open(a) as ai, Image.open(b) as bi:
        x = np.asarray(ai.convert("L").resize((128, 128)), dtype=np.float32)
        y = np.asarray(bi.convert("L").resize((128, 128)), dtype=np.float32)
    ux, uy = float(x.mean()), float(y.mean())
    vx, vy = float(x.var()), float(y.var())
    cov = float(((x - ux) * (y - uy)).mean())
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return ((2 * ux * uy + c1) * (2 * cov + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))


def dedup_and_sample(cfg: dict, milvus=None) -> dict:
    frames = read_jsonl(output(cfg, "derived", "frames.jsonl"))
    if not frames:
        raise RuntimeError("inventory has no valid frames")
    by_video = defaultdict(list)
    for row in frames:
        by_video[row["video_id"]].append(row)
    exact = {}
    groups, sampled, reasons = [], [], Counter()
    rng = random.Random(cfg["seed"])
    representatives = {}
    for video, rows in sorted(by_video.items()):
        rows.sort(key=lambda r: r["n"])
        vectors, ocr, metadata = {}, {}, {}
        if milvus is not None:
            ids = [r["frame_id"] for r in rows]
            for start in range(0, len(ids), 200):
                batch = ids[start:start + 200]
                vectors.update(milvus.get_vectors(batch, "text_vector"))
                metadata.update({r["id"]: r for r in milvus.get_by_ids(batch)})
                ocr.update({fid: metadata[fid].get("ocr", "").strip().casefold()
                            for fid in batch if fid in metadata})
        current = None
        for row in rows:
            digest = row["sha256"]
            if digest in exact:
                gid = exact[digest]
                reasons["exact"] += 1
            else:
                gid = row["frame_id"]
                if current is not None:
                    rep = representatives[current["group_id"]]
                    distance = (int(row["phash"], 16) ^ int(rep["phash"], 16)).bit_count()
                    if distance <= cfg["sampling"]["phash_distance"]:
                        sim = _ssim(Path(rep["image_path"]), Path(row["image_path"]))
                        text_unknown = not ocr.get(rep["frame_id"]) and not ocr.get(row["frame_id"])
                        required_ssim = max(0.995, cfg["sampling"]["ssim_min"]) if text_unknown else cfg["sampling"]["ssim_min"]
                        if sim >= required_ssim and milvus is not None:
                            a, b = vectors.get(rep["frame_id"]), vectors.get(row["frame_id"])
                            if a is not None and b is not None:
                                cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                                # A near merge requires positive OCR equivalence; absence is not proof.
                                if cosine >= cfg["sampling"]["cosine_min"] and ocr.get(rep["frame_id"]) == ocr.get(row["frame_id"]):
                                    gid = current["group_id"]
                                    reasons["near"] += 1
                exact[digest] = gid
            representatives.setdefault(gid, row)
            row["group_id"] = gid
            groups.append({"frame_id": row["frame_id"], "near_duplicate_group_id": gid})
            current = row
        candidates = [r for r in rows if r["frame_id"] == r["group_id"] and r["video_path"]
                      and (milvus is None or r["frame_id"] in vectors)]
        for candidate in candidates:
            fid = candidate["frame_id"]
            candidate["ocr_present"] = bool(ocr.get(fid))
            candidate["object_labels"] = list(metadata.get(fid, {}).get("objects", []))
        chosen = []
        if milvus is not None:
            ocr_candidates = [r for r in candidates if ocr.get(r["frame_id"])]
            if ocr_candidates:
                chosen.append(rng.choice(ocr_candidates))
            object_candidates = sorted(candidates, key=lambda r: len(set(metadata.get(r["frame_id"], {}).get("objects", []))), reverse=True)
            if object_candidates and object_candidates[0] not in chosen:
                chosen.append(object_candidates[0])
        rest = [r for r in candidates if r not in chosen]
        # Randomize inside broad temporal bins to avoid a run of adjacent frames.
        rest.sort(key=lambda r: (r["n"] // 30, rng.random()))
        while rest and len(chosen) < cfg["sampling"]["max_per_video"]:
            pick = rest.pop(0)
            chosen.append(pick)
            rest = [r for r in rest if abs(r["n"] - pick["n"]) >= 10]
        sampled.extend(chosen)
    rng.shuffle(sampled)
    sampled = sampled[:cfg["sampling"]["max_proposals"]]
    write_jsonl(output(cfg, "derived", "near_duplicates.jsonl"), groups)
    write_jsonl(output(cfg, "derived", "sampled_frames.jsonl"), sampled)
    report = {"frames": len(frames), "groups": len(set(r["near_duplicate_group_id"] for r in groups)),
              "sampled": len(sampled), "merge_reasons": dict(reasons),
              "sampled_videos": len({r["video_id"] for r in sampled}),
              "sampled_ocr": sum(r.get("ocr_present", False) for r in sampled),
              "sampled_with_objects": sum(bool(r.get("object_labels")) for r in sampled),
              "near_dedup_enabled": milvus is not None}
    write_json(output(cfg, "derived", "sample_report.json"), report)
    return report


def verify_siglip2(cfg: dict, sample_size: int = 3) -> dict:
    import transformers
    from backend.app.encoder import SigLIP2Encoder
    from backend.app.milvus_client import MilvusSearchService
    frames = read_jsonl(output(cfg, "derived", "frames.jsonl"))
    if not frames:
        raise RuntimeError("inventory required before index verification")
    client = MilvusSearchService(cfg["milvus"]["uri"], cfg["milvus"]["collection"])
    encoder = SigLIP2Encoder(cfg["milvus"]["siglip2_model"])
    sample = random.Random(cfg["seed"]).sample(frames, min(sample_size, len(frames)))
    indexed = client.get_vectors([r["frame_id"] for r in sample], "text_vector")
    metadata = {r["id"]: r for r in client.get_by_ids([r["frame_id"] for r in sample])}
    details = []
    for row in sample:
        with Image.open(row["image_path"]) as image:
            fresh = encoder.encode_image(image.convert("RGB"))
        old = indexed.get(row["frame_id"])
        cosine = float(np.dot(fresh, old) / (np.linalg.norm(fresh) * np.linalg.norm(old))) if old is not None else None
        mapped = metadata.get(row["frame_id"], {})
        alignment = mapped.get("frame_idx") == row["frame_idx"] and abs(float(mapped.get("pts_time", -999)) - row["pts_time"]) < 0.001
        details.append({"frame_id": row["frame_id"], "cosine": cosine, "map_aligned": alignment})
    model_config = encoder._model.config.to_json_string()
    processor_config = str(encoder._processor)
    report = {"model": cfg["milvus"]["siglip2_model"],
              "transformers_version": transformers.__version__,
              "model_config_sha256": hashlib.sha256(model_config.encode()).hexdigest(),
              "processor_config_sha256": hashlib.sha256(processor_config.encode()).hexdigest(),
              "samples": details,
              "passed": all(r["cosine"] is not None and r["cosine"] >= 0.995 and r["map_aligned"] for r in details)}
    write_json(output(cfg, "derived", "index_verification.json"), report)
    if not report["passed"]:
        raise RuntimeError("SigLIP2 sample parity failed; pilot must stop before retrieval")
    return report


def reembed(cfg: dict, batch_size: int = 16) -> dict:
    """Encode original keyframes with the configured checkpoint into versioned .npy files.

    This does not mutate the Milvus collection. It supports rebuilding an index
    after a parity failure, with its own checkpointed per-video outputs.
    """
    from backend.app.encoder import SigLIP2Encoder
    frames = read_jsonl(output(cfg, "derived", "frames.jsonl"))
    if not frames:
        raise RuntimeError("inventory required before re-embedding")
    encoder = SigLIP2Encoder(cfg["milvus"]["siglip2_model"])
    by_video = defaultdict(list)
    for row in frames:
        by_video[row["video_id"]].append(row)
    written = 0
    for video, rows in sorted(by_video.items()):
        dest = output(cfg, "derived", "siglip2_features", video + ".npy")
        if dest.exists():
            continue
        rows.sort(key=lambda r: r["n"])
        vectors = []
        for row in rows:
            with Image.open(row["image_path"]) as image:
                vectors.append(encoder.encode_image(image.convert("RGB")))
        np.save(dest, np.asarray(vectors, dtype=np.float32))
        write_json(output(cfg, "derived", "siglip2_features", video + ".json"),
                   {"model": cfg["milvus"]["siglip2_model"], "frame_ids": [r["frame_id"] for r in rows],
                    "dim": len(vectors[0]) if vectors else 0})
        written += 1
    return {"videos_written": written, "videos_total": len(by_video)}

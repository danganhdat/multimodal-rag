import json
from pathlib import Path

VIDEO_META: dict[str, dict] = {}
ALL_OBJECT_CLASSES: set[str] = set()


def load_video_metadata(media_info_dir: Path) -> None:
    VIDEO_META.clear()
    for json_file in sorted(media_info_dir.glob("*.json")):
        video_name = json_file.stem
        try:
            VIDEO_META[video_name] = json.loads(
                json_file.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass


def get_video_meta(video_name: str) -> dict | None:
    return VIDEO_META.get(video_name)

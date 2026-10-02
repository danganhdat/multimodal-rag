import json

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..video_metadata import VIDEO_META

router = APIRouter()


@router.get("/video-meta/{video_name}")
async def get_video_meta(video_name: str):
    meta = VIDEO_META.get(video_name)
    if meta is None:
        raise HTTPException(404, "Video metadata not found")
    return meta


@router.get("/object-classes")
async def get_object_classes():
    obj_path = settings.DATA_ROOT / "object_classes.json"
    if obj_path.exists():
        classes = json.loads(obj_path.read_text(encoding="utf-8"))
    else:
        classes = []
    return {"classes": classes}

import os
import re

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse

from ..config import settings

router = APIRouter()


@router.get("/video/{video_name}")
async def get_video(video_name: str, request: Request):
    video_path = settings.VIDEOS_DIR / f"{video_name}.mp4"
    if not video_path.exists():
        raise HTTPException(404, "Video not found")

    file_size = video_path.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        match = re.match(r"bytes=(\d+)-(\d*)", range_header)
        if not match:
            raise HTTPException(416, "Invalid range")
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else file_size - 1
        end = min(end, file_size - 1)
        length = end - start + 1

        with open(video_path, "rb") as f:
            f.seek(start)
            data = f.read(length)

        return Response(
            content=data,
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
            },
        )

    return FileResponse(video_path, media_type="video/mp4")

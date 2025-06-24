from fastapi import FastAPI, HTTPException, Response
from typing import List
import os
import mimetypes
from collections import defaultdict

app = FastAPI()

IMAGES_DIR = "../data/keyframes"
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')


def is_valid_image(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTENSIONS)


@app.get("/keyframes", response_model=List[List[str]])
async def list_images():
    images_by_folder = defaultdict(list)

    for root, _, files in os.walk(IMAGES_DIR):
        for filename in files:
            if is_valid_image(filename):
                rel_path = os.path.relpath(os.path.join(root, filename), IMAGES_DIR)
                rel_path = rel_path.replace(os.sep, "/")

                folder = rel_path.split("/", 1)[0] if "/" in rel_path else ""
                images_by_folder[folder].append(rel_path)

    return list(images_by_folder.values())


@app.get("/keyframes/{filename:path}")
async def get_image(filename: str):
    file_path = os.path.abspath(os.path.join(IMAGES_DIR, filename))

    if not file_path.startswith(os.path.abspath(IMAGES_DIR)):
        raise HTTPException(status_code=403, detail="Access denied.")

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Image not found.")

    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type = mime_type or "application/octet-stream"

    with open(file_path, "rb") as image_file:
        return Response(content=image_file.read(), media_type=mime_type)

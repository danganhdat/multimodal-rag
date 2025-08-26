from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pymilvus import AnnSearchRequest, RRFRanker, MilvusClient, Function, FunctionType
from typing import List, Optional
from pydantic import BaseModel
from typing import List
from encoder import SigLIPEncoder, TaskFormerEncoder

import pyjokes
import os
import base64
import datetime

# --- CONFIG ---
COLLECTION_NAME = "my_collection"
IMAGES_DIR = os.path.abspath("H:/aic25-batch1-keyframes/keyframes")
SKETCHES_DIR = os.path.abspath("../data/sketches")

# --- APP INIT ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sig = SigLIPEncoder()
task = TaskFormerEncoder()
ranker = RRFRanker(100)
rerank = Function(
        name="weight",
        input_field_names=[], # Must be an empty list
        function_type=FunctionType.RERANK,
        params={
            "reranker": "weighted", 
            "weights": [0.2, 0.8],
            "norm_score": True  # Optional
        }
    )
client = MilvusClient(uri="http://localhost:19530")

if not client.has_collection("my_collection"):
    print(f"[Milvus] {COLLECTION_NAME} not found.")

# --- MODELS ---

class Address(BaseModel):
    folder_name: str
    image_name: str

class SurroundingsRequest(BaseModel):
    addresses: List[Address]
    window: int = 25
    
# --- ROUTES ---

@app.get("/")
async def root():
    random_joke = pyjokes.get_joke("en", "neutral")
    return {"random_joke": random_joke}

class SearchSketchRequest(BaseModel):
    sketch_path: str              # path to sketch image
    text: str                     # optional text query (default empty)

class SearchRequest(BaseModel):
    queries: str                  # main text query
    sketch: Optional[SearchSketchRequest] = None
    ocr: List[str] = []
    objects: List[str] = []
    colours: List[str] = []
    limit: int = 50


@app.post("/search")
def search(request: SearchRequest) -> List[dict]:

    filter_parts = []
    if request.ocr:
        for ocr in request.ocr:
            filter_parts.append(f'ocr like \"%{ocr}%\"')
    if request.objects:
        filter_parts.append(f'ARRAY_CONTAINS_ANY(objects, {request.objects})')
    if request.colours:
        filter_parts.append(f'ARRAY_CONTAINS_ANY(colours, {request.colours})')
    filters = " AND ".join(filter_parts) if filter_parts else None

    results = None

    # text_embedding = sig.encode_text(request.queries)
    # text_search = client.search(
    #     collection_name=COLLECTION_NAME,
    #     data=[text_embedding],
    #     anns_field="text_dense",
    #     limit=request.limit,
    #     filter=filters,
    #     output_fields=["path", "ocr", "objects", "colours"],
    # )
    
    if request.sketch and request.sketch.text:
        fuse_embedding = task.get_feature(request.sketch.sketch_path, request.sketch.text)

        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[fuse_embedding],
            anns_field="sketch_dense",
            limit=request.limit,
            filter=filters,
            output_fields=["path", "ocr", "objects", "colours"],
        )

        # results = client.hybrid_search(
        #     collection_name=COLLECTION_NAME,
        #     reqs=[text_search, sketch_search],
        #     ranker=ranker,
        #     limit=request.limit,
        #     output_fields=["path", "ocr", "objects", "colours"],
        # )
        # results = [text_search, sketch_search]

    else:
        raise ValueError("Invalid search request")

    hits: List[dict] = []
    for hit in results[0]:
        hits.append({
            "path": hit.get("path"),
            "ocr": hit.get("ocr", ""),
            "objects": list(hit.get("objects", [])),
            "colours": list(hit.get("colours", []))
        })
    return hits


@app.post("/surroundings", response_model=List[Address])
def get_surroundings(request: SurroundingsRequest):
    """
    For each address, return a window of nearby keyframes.
    """
    results = []
    for addr in request.addresses:
        folder_path = os.path.join(IMAGES_DIR, addr.folder_name)
        if not os.path.exists(folder_path):
            results.append([])
            continue
        files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])
        if addr.image_name not in files:
            results.append([])
            continue
        idx = files.index(addr.image_name)
        start, end = max(0, idx - request.window), min(len(files), idx + request.window + 1)
        group = [
            Address(folder_name=addr.folder_name, image_name=files[i])
            for i in range(start, end)
        ]
        results.append(group)
    return results

@app.get("/image/{folder_name}/{image_name}")
def get_image(folder_name: str, image_name: str):
    """
    Serve an image file by URL.
    """
    path = os.path.join(IMAGES_DIR, folder_name, image_name)
    if not os.path.exists(path):
        raise HTTPException(404, "Image not found")
    return FileResponse(path)
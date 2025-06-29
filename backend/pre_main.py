
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pymilvus import AnnSearchRequest, RRFRanker

from pymilvus import MilvusClient

from pydantic import BaseModel
from typing import List
from clip_encoder import ClipEncoder

import os
import pickle
import base64

COLLECTION_NAME = "clip_images"
IMAGES_DIR = os.path.abspath("../data/keyframes")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
encoder = ClipEncoder()
milvus_client = MilvusClient(uri="http://localhost:19530")

if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(COLLECTION_NAME)

milvus_client.create_collection(
    collection_name=COLLECTION_NAME,
    dimension=512,
    auto_id=True
)

for i in range(9):
    with open(f"embeddings/aic_2023_clip_{i}.pkl", 'rb') as f:
        data = pickle.load(f)
    milvus_client.insert(collection_name=COLLECTION_NAME, data=data)

class MultiSearchRequest(BaseModel):
    queries: List[str]
    limit: int = 10

class Address(BaseModel):
    folder_name: str
    image_name: str
    
class SurroundingsRequest(BaseModel):
    addresses: List[Address]
    window: int = 100

@app.get("/")
async def get_hello():
    return {"message": "Hello mother fucker!"}


@app.post("/search", response_model=List[List[Address]])
def search(request: MultiSearchRequest):
    all_results = []
    for query in request.queries:
        embedding = encoder.encode_text(query)
        results = milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[embedding],
            limit=request.limit,
            output_fields=["filepath"],
            params={"metric_type": "L2", "params": {}}
        )
        addresses = []
        for result in results[0]:
            folder, image = result.get("entity").get("filepath").split('/', 1)
            addresses.append(Address(folder_name=folder, image_name=image))
        all_results.append(addresses)
    return all_results

@app.post("/surroundings", response_model=List[List[Address]])
def get_surroundings(request: SurroundingsRequest):
    results = []
    for addr in request.addresses:
        folder_path = os.path.join(IMAGES_DIR, addr.folder_name)
        if not os.path.exists(folder_path):
            results.append([])
            continue

        # Only get image files, sort alphabetically
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

@app.post("/hybrid_search", response_model=List[Address])
def search(request: MultiSearchRequest):
    """
    Hybrid multi-query search using Milvus with RRF ranking.
    Returns a list of unique Address objects.
    """
    # Prepare ANN requests for each query
    reqs = [
        AnnSearchRequest(
            data=[encoder.encode_text(query)],
            anns_field="vector",          # Make sure this matches your schema
            param={},      # Or your search index param
            limit=request.limit
        )
        for query in request.queries
    ]

    # Perform the hybrid search with rank fusion
    res = milvus_client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=reqs,
        ranker=ranker,
        limit=request.limit,
        output_fields=["filepath"]
    )

    # Parse results and build Address list
    addresses = []
    seen = set()
    if res and res[0]:
        for hit in res[0]:  # With RRF, all hits are fused into the first list
            # Defensive: Milvus result may be object or dict
            filepath = getattr(hit.entity, "filepath", None) if hasattr(hit, "entity") else hit.get("entity", {}).get("filepath")
            if not filepath or filepath in seen:
                continue
            if "/" not in filepath:
                continue  # Defensive: skip bad paths
            folder, image = filepath.split("/", 1)
            addresses.append(Address(folder_name=folder, image_name=image))
            seen.add(filepath)
            if len(addresses) >= request.limit:
                break
    return addresses

@app.get("/image/{folder}/{filename}")
def get_image(folder: str, filename: str):
    path = os.path.join(IMAGES_DIR, folder, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Image not found")
    return FileResponse(path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
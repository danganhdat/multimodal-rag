from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pymilvus import AnnSearchRequest, RRFRanker, MilvusClient

from pydantic import BaseModel
from typing import List, Optional
from clip_encoder import ClipEncoder

import os
import pickle

# --- CONFIG ---
COLLECTION_NAME = "clip_images"
IMAGES_DIR = os.path.abspath("D:\\SGU_University\\Project_ca_nhan\\AIC_BE\\data\\keyframes")

# --- APP INIT ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

encoder = ClipEncoder()
milvus_client = MilvusClient(uri="http://localhost:19530")
ranker = RRFRanker(100)

# --- MILVUS SETUP (ONE-TIME) ---
if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(COLLECTION_NAME)
milvus_client.create_collection(
    collection_name=COLLECTION_NAME,
    dimension=512,
    auto_id=True
)
for i in range(1):
    with open(f"embeddings/aic_2023_clip_{i}.pkl", "rb") as f:
        data = pickle.load(f)
    milvus_client.insert(collection_name=COLLECTION_NAME, data=data)

# --- MODELS ---
class Address(BaseModel):
    folder_name: str
    image_name: str

class SearchRequest(BaseModel):
    queries: List[str]
    limit: int = 10

class SurroundingsRequest(BaseModel):
    addresses: List[Address]
    window: int = 10

# --- ROUTES ---

@app.get("/")
async def root():
    return {"message": "Hello world!"}

@app.post("/search", response_model=List[Address])
def search(request: SearchRequest):
    """
    Search for keyframes by a single query.
    """
    embedding = encoder.encode_text(request.queries[0])
    results = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[embedding],
        limit=request.limit,
        output_fields=["filepath"],
        params={"metric_type": "L2", "params": {}}
    )
    addresses = []
    for result in results[0]:
    # for hit in result:
        filepath = result["entity"]["filepath"]
        if filepath and "/" in filepath:
            folder, image = filepath.split("/", 1)
            addresses.append(Address(folder_name=folder, image_name=image))
    return addresses

@app.post("/search_hybrid", response_model=List[Address])
def search_hybrid(request: SearchRequest):
    """
    Perform a hybrid (multi-query) search using Reciprocal Rank Fusion (RRF).
    Returns up to `limit` unique Address objects, ranked by fusion.
    """
    # Prepare an AnnSearchRequest for each query string
    reqs = [
        AnnSearchRequest(
            data=[encoder.encode_text(query)],
            anns_field="vector",   # Make sure your Milvus schema matches this field
            param={},
            limit=request.limit
        )
        for query in request.queries
    ]
    # Run hybrid search with rank fusion
    res = milvus_client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=reqs,
        ranker=ranker,
        limit=request.limit,
        output_fields=["filepath"]
    )

    addresses = []
    seen = set()
    # Defensive: Check for results, iterate over hits
    hits = res[0] if res and len(res) > 0 else []
    for hit in hits:
        # Safely extract filepath from entity (object or dict)
        if hasattr(hit, "entity") and hasattr(hit.entity, "filepath"):
            filepath = hit.entity.filepath
        else:
            filepath = hit.get("entity", {}).get("filepath")
        if filepath and "/" in filepath and filepath not in seen:
            folder, image = filepath.split("/", 1)
            addresses.append(Address(folder_name=folder, image_name=image))
            seen.add(filepath)
        if len(addresses) >= request.limit:
            break

    return addresses

@app.post("/surroundings", response_model=List[List[Address]])
def get_surroundings(request: SurroundingsRequest):
    """
    For each address, return a window of nearby keyframes.
    """
    print(f"Requesting surroundings for {len(request.addresses)} addresses with window size {request.window}")
    results = []
    for addr in request.addresses:
        print(f"[DEBUG] Processing address: {addr.folder_name}/{addr.image_name}")
        folder_path = os.path.join(IMAGES_DIR, addr.folder_name)
        print(f"[DEBUG] Full folder path: {folder_path}")
        print(f"[DEBUG] IMAGES_DIR: {IMAGES_DIR}")
        print(f"[DEBUG] Folder exists: {os.path.exists(folder_path)}")
        
        if not os.path.exists(folder_path):
            print(f"[DEBUG] Folder not found, returning empty list")
            results.append([])
            continue
            
        files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])
        print(f"[DEBUG] Found {len(files)} image files in folder")
        print(f"[DEBUG] First 5 files: {files[:5] if files else 'None'}")
        print(f"[DEBUG] Looking for image: {addr.image_name}")
        print(f"[DEBUG] Image found in list: {addr.image_name in files}")
        
        if addr.image_name not in files:
            print(f"[DEBUG] Image not found in files list, returning empty list")
            results.append([])
            continue
            
        idx = files.index(addr.image_name)
        start, end = max(0, idx - request.window), min(len(files), idx + request.window + 1)
        print(f"[DEBUG] Image index: {idx}, window: {start}-{end}")
        
        group = [
            Address(folder_name=addr.folder_name, image_name=files[i])
            for i in range(start, end)
        ]
        print(f"[DEBUG] Created group with {len(group)} images")
        results.append(group)
    
    print(f"[DEBUG] Final results: {len(results)} groups")
    return results

@app.get("/image/{folder_name}/{image_name}")
def get_image(folder_name: str, image_name: str):
    """
    Serve an image file by URL.
    """
    print(f"Requesting image: {folder_name}/{image_name}")
    path = os.path.join(IMAGES_DIR, folder_name, image_name)
    if not os.path.exists(path):
        raise HTTPException(404, "Image not found")
    return FileResponse(path)

# --- MAIN ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

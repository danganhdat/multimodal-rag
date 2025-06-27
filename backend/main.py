from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os
import pickle
from clip_encoder import ClipEncoder
from pymilvus import MilvusClient, connections

# --- Configuration ---
COLLECTION_NAME = "clip_image_collection"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# IMAGES_DIR = "../data/keyframes"
IMAGES_DIR = os.path.abspath("../data/keyframes")  # Always use absolute path!
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')

# --- Initialize FastAPI ---
app = FastAPI()

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize components ---
encoder = ClipEncoder()
milvus_client = MilvusClient(uri="http://localhost:19530")

# --- Milvus Collection Initialization ---
if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(COLLECTION_NAME)

milvus_client.create_collection(
    collection_name=COLLECTION_NAME,
    dimension=512,
    auto_id=True,
    enable_dynamic_field=True,
)

# Perform insertion and capture response
for i in range(18):
    chunk_filename = f"embeddings/aic_2023_clip_{i+1}.pkl"
    with open(chunk_filename, 'rb') as f:
        chunk_data = pickle.load(f)

    insert_result = milvus_client.insert(
        collection_name=COLLECTION_NAME,
        data=chunk_data
    )

print(milvus_client.get_collection_stats(COLLECTION_NAME))

# --- Request Models ---
class SearchRequest(BaseModel):
    query: str
    limit: int = 10

# --- API Routes ---
@app.get("/")
async def get_hello():
    return {"message": "Hello, welcome to the CLIP Image Search API!"}

@app.get("/image/{rel_path:path}")
async def get_image(rel_path: str):
    images_dir = os.path.normpath(IMAGES_DIR)
    safe_path = os.path.normpath(os.path.abspath(os.path.join(images_dir, rel_path)))

    if not safe_path.startswith(IMAGES_DIR):
        raise HTTPException(status_code=403, detail="Access denied.")
    if not os.path.isfile(safe_path):
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(safe_path)

@app.post("/search/")
async def search_images(req: SearchRequest):
    query_embedding = encoder.encode_text(req.query)

    search_results = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        limit=req.limit,
        output_fields=["filepath"],
    )
    
    results = []
    for result in search_results:
        for hit in result:
            filepath = hit["entity"]["filepath"]
            results.append(filepath)
    return JSONResponse({"filepath ": results})

# --- Entry Point ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)
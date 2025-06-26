from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os
import pickle
from clip_encoder import ClipEncoder
from pymilvus import MilvusClient, connections

connections.connect(
    alias="default",
    uri="http://localhost:19530",
)

# --- Configuration ---
COLLECTION_NAME = "clip_image_collection"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "keyframes"))
print(BASE_DIR)
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

with open("aic_2023_clip.pkl", 'rb') as f:
    clip_embed_data = pickle.load(f)

# Perform insertion and capture response
for i in range(1):
    chunk_filename = f"aic_2023_clip_{i+1}.pkl"
    with open(chunk_filename, 'rb') as f:
        chunk_data = pickle.load(f)

    insert_result = milvus_client.insert(
        collection_name=COLLECTION_NAME,
        data=chunk_data
    )
print(chunk_data[0])
# Check insert result for IDs or status
if insert_result.get("insert_count", 0) > 0:
    print(f"Successfully inserted {insert_result['insert_count']} items.")
else:
    print("Insertion failed or no data inserted.")

# --- Request Models ---
class SearchRequest(BaseModel):
    query: str
    limit: int = 10

# --- API Routes ---
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
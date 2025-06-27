from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os
import pickle
from clip_encoder import ClipEncoder
from pymilvus import MilvusClient

# --- Configuration ---
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "clip_image_collection")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_URI = f"http://{MILVUS_HOST}:{MILVUS_PORT}"

# Construct IMAGES_DIR relative to this file's directory
# Assumes 'data/keyframes' is one level up from 'backend' directory
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.abspath(os.path.join(CURRENT_SCRIPT_DIR, "../data/keyframes"))
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp') # Unused, consider removing if not needed elsewhere

# --- Initialize FastAPI ---
app = FastAPI()

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods
    allow_headers=["*"], # Allow all headers
)

# --- Initialize components ---
try:
    encoder = ClipEncoder()
except Exception as e:
    print(f"Error initializing ClipEncoder: {e}")
    # Potentially exit or raise if encoder is critical for startup
    raise

try:
    print(f"Connecting to Milvus at {MILVUS_URI}...")
    milvus_client = MilvusClient(uri=MILVUS_URI)
except Exception as e:
    print(f"Error connecting to Milvus: {e}")
    # Potentially exit or raise if Milvus is critical
    raise

# --- Milvus Collection and Data Initialization ---
def initialize_milvus_data():
    print(f"Checking Milvus collection: {COLLECTION_NAME}")
    collection_exists = milvus_client.has_collection(COLLECTION_NAME)

    if not collection_exists:
        print(f"Collection '{COLLECTION_NAME}' does not exist. Creating...")
        milvus_client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=512, # Assuming CLIP model output dimension
            auto_id=True,
            enable_dynamic_field=True,
            description="CLIP Image Embeddings"
        )
        print(f"Collection '{COLLECTION_NAME}' created.")
        # Proceed to load data since it's a new collection
        load_data = True
    else:
        print(f"Collection '{COLLECTION_NAME}' exists.")
        # Check if collection is empty
        stats = milvus_client.get_collection_stats(COLLECTION_NAME)
        num_entities = 0
        for stat_pair in stats:
            if stat_pair.key == "row_count":
                num_entities = int(stat_pair.value) # Milvus returns string value for stats
                break

        if num_entities == 0:
            print(f"Collection '{COLLECTION_NAME}' is empty. Loading data...")
            load_data = True
        else:
            print(f"Collection '{COLLECTION_NAME}' already contains {num_entities} entities. Skipping data loading.")
            load_data = False

    if load_data:
        print("Starting data insertion...")
        # TODO: Make the number of chunks configurable or discoverable
        for i in range(18): # Assuming 18 embedding chunks
            chunk_filename = os.path.join(CURRENT_SCRIPT_DIR, f"embeddings/aic_2023_clip_{i+1}.pkl")
            if not os.path.exists(chunk_filename):
                print(f"Warning: Embedding chunk file not found: {chunk_filename}. Skipping.")
                continue
            try:
                with open(chunk_filename, 'rb') as f:
                    chunk_data = pickle.load(f)

                if not chunk_data:
                    print(f"Warning: Embedding chunk {chunk_filename} is empty. Skipping.")
                    continue

                print(f"Inserting data from {chunk_filename}...")
                insert_result = milvus_client.insert(
                    collection_name=COLLECTION_NAME,
                    data=chunk_data
                )
                print(f"Successfully inserted {len(insert_result.primary_keys)} entities from {chunk_filename}.")
            except FileNotFoundError:
                print(f"Error: Embedding file {chunk_filename} not found.")
            except pickle.UnpicklingError:
                print(f"Error: Could not unpickle {chunk_filename}.")
            except Exception as e:
                print(f"An error occurred while inserting data from {chunk_filename}: {e}")

        print("Data insertion process completed.")
        final_stats = milvus_client.get_collection_stats(COLLECTION_NAME)
        print(f"Final collection stats for '{COLLECTION_NAME}': {final_stats}")

# Call initialization at startup
initialize_milvus_data()

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
        data=[query_embedding], # Search expects a list of embeddings
        limit=req.limit,
        output_fields=["filepath"], # Ensure 'filepath' is stored during insertion or is a dynamic field
    )
    
    # Simplify result extraction
    # Each `search_result` in `search_results` corresponds to a query vector. We have one.
    # Each `hit` in `search_result` is a document found.
    filepaths = [hit["entity"]["filepath"] for hit in search_results[0]] if search_results and search_results[0] else []

    return JSONResponse(content={"filepaths": filepaths})

# --- Entry Point ---
if __name__ == "__main__":
    import uvicorn
    # Corrected to use 'main:app' as this file is main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
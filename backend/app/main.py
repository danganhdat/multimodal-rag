from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .encoder import CLIPEncoder
from .milvus_client import MilvusSearchService
from .reranker import Reranker
from .video_metadata import load_video_metadata


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading CLIP encoder...")
    app.state.encoder = CLIPEncoder(settings.CLIP_MODEL)
    print("Loading reranker...")
    try:
        app.state.reranker = Reranker()
    except Exception as e:
        print(f"Reranker not available: {e}")
        app.state.reranker = None
    print("Connecting to Milvus...")
    app.state.milvus = MilvusSearchService(settings.MILVUS_URI, settings.COLLECTION_NAME)
    print("Loading video metadata...")
    load_video_metadata(settings.MEDIA_INFO_DIR)
    print("Ready.")
    yield


app = FastAPI(title="AIC25 Multimodal Search", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from .routes import media, metadata, search, translate

app.include_router(search.router, prefix="/api")
app.include_router(media.router, prefix="/api")
app.include_router(translate.router, prefix="/api")
app.include_router(metadata.router, prefix="/api")

if settings.KEYFRAMES_DIR.exists():
    app.mount("/static/keyframes", StaticFiles(directory=settings.KEYFRAMES_DIR), name="keyframes")

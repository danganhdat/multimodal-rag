from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .encoder import CLIPEncoder, SigLIP2Encoder
from .milvus_client import MilvusSearchService
from .video_metadata import load_video_metadata


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"DATA_ROOT:    {settings.DATA_ROOT.resolve()}")
    print(f"  keyframes:  {settings.KEYFRAMES_DIR}")
    print(f"  videos:     {settings.VIDEOS_DIR}")
    print(f"  checkpoint: {settings.DATA_ROOT / settings.CLIP_CHECKPOINT}")
    print("CLIP encoder ready (lazy-load)...")
    app.state.encoder = CLIPEncoder(settings.CLIP_MODEL, str(settings.DATA_ROOT / settings.CLIP_CHECKPOINT))
    print("SigLIP2 encoder ready (lazy-load)...")
    app.state.siglip2_encoder = SigLIP2Encoder(settings.SIGLIP2_MODEL)
    print("Loading reranker...")
    try:
        from .reranker import Reranker
        app.state.reranker = Reranker(settings.RERANKER_MODEL)
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


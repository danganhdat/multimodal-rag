import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .encoder import SketchEncoder, SigLIP2Encoder
from .milvus_client import MilvusSearchService
from .video_metadata import load_video_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = _PROJECT_ROOT / settings.CLIP_CHECKPOINT
    logger.info("DATA_ROOT:    %s", settings.DATA_ROOT.resolve())
    logger.info("  keyframes:  %s", settings.KEYFRAMES_DIR)
    logger.info("  videos:     %s", settings.VIDEOS_DIR)
    logger.info("  checkpoint: %s", checkpoint)
    logger.info("Sketch encoder ready (lazy-load)")
    app.state.sketch_encoder = SketchEncoder(settings.CLIP_MODEL, str(checkpoint))
    logger.info("SigLIP2 encoder ready (lazy-load)")
    app.state.siglip2_encoder = SigLIP2Encoder(settings.SIGLIP2_MODEL)
    logger.info("Connecting to Milvus...")
    app.state.milvus = MilvusSearchService(settings.MILVUS_URI, settings.COLLECTION_NAME)
    logger.info("Loading video metadata...")
    load_video_metadata(settings.MEDIA_INFO_DIR)

    logger.info("Building BM25 index...")
    try:
        from .retrieve.bm25 import BM25Index
        bm25 = BM25Index()
        bm25.build(app.state.milvus.iter_ocr())
        app.state.bm25_index = bm25
        if bm25.ready:
            logger.info("BM25 index built: %d documents", bm25.n_docs)
        else:
            logger.warning("BM25 index: no OCR data found (disabled)")
    except Exception as e:
        logger.exception("BM25 index failed")
        app.state.bm25_index = None

    logger.info("Ready.")
    yield


app = FastAPI(title="AIC25 VQA Search", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from .middleware import TraceMiddleware
app.add_middleware(TraceMiddleware)

from .routes import media, metadata, search, translate
from .routes.v1 import search as v1_search
from .routes.v1 import query as v1_query
from .routes.v1 import ask as v1_ask
from .routes.v1 import chat as v1_chat
from .routes.v1 import feedback as v1_feedback
from .routes.v1 import traces as v1_traces
from .routes.v1 import answers as v1_answers
from .routes.v1 import stats as v1_stats
from .routes.v1 import vqa as v1_vqa

app.include_router(search.router, prefix="/api")
app.include_router(media.router, prefix="/api")
app.include_router(translate.router, prefix="/api")
app.include_router(metadata.router, prefix="/api")
app.include_router(v1_search.router, prefix="/api/v1")
app.include_router(v1_query.router, prefix="/api/v1")
app.include_router(v1_ask.router, prefix="/api/v1")
app.include_router(v1_chat.router, prefix="/api/v1")
app.include_router(v1_feedback.router, prefix="/api/v1")
app.include_router(v1_traces.router, prefix="/api/v1")
app.include_router(v1_answers.router, prefix="/api/v1")
app.include_router(v1_stats.router, prefix="/api/v1")
app.include_router(v1_vqa.router, prefix="/api/v1")


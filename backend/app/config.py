from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATA_ROOT: Path = Path("C:/Users/datda/Works/multimodal-rag")
    MILVUS_URI: str = "http://localhost:19530"
    COLLECTION_NAME: str = "aic25_keyframes"
    CLIP_MODEL: str = "sentence-transformers/clip-ViT-B-32"
    OBJECT_SCORE_THRESHOLD: float = 0.3
    OBJECT_MAX_PER_FRAME: int = 20
    IMAGE_BASE_URL: str = "/static/keyframes"

    @property
    def KEYFRAMES_DIR(self) -> Path:
        return self.DATA_ROOT / "keyframes"

    @property
    def CLIP_FEATURES_DIR(self) -> Path:
        return self.DATA_ROOT / "clip-features-32"

    @property
    def MAP_KEYFRAMES_DIR(self) -> Path:
        return self.DATA_ROOT / "map-keyframes"

    @property
    def OBJECTS_DIR(self) -> Path:
        return self.DATA_ROOT / "objects"

    @property
    def MEDIA_INFO_DIR(self) -> Path:
        return self.DATA_ROOT / "media-info"

    @property
    def VIDEOS_DIR(self) -> Path:
        return self.DATA_ROOT / "videos"

    model_config = {
        "env_file": Path(__file__).resolve().parent.parent.parent / ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()

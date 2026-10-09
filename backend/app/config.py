from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATA_ROOT: Path = Path(".")
    MILVUS_URI: str = "http://localhost:19530"
    COLLECTION_NAME: str = "aic25_keyframes"
    CLIP_MODEL: str = "ViT-B-16"
    CLIP_CHECKPOINT: str = "tsbir_model_weight/tsbir_model_final.pt"
    SIGLIP2_MODEL: str = "google/siglip2-base-patch16-512"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    OBJECT_SCORE_THRESHOLD: float = 0.3
    OBJECT_MAX_PER_FRAME: int = 20
    IMAGE_BASE_URL: str = "/keyframes"
    GCP_PROJECT_ID: str = ""
    GOOGLE_APPLICATION_CREDENTIALS: str = ""

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

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

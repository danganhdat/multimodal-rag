from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATA_ROOT: Path = Path(".")
    MILVUS_URI: str = "http://localhost:19530"
    COLLECTION_NAME: str = "aic25_keyframes"
    CLIP_MODEL: str = "ViT-B-16"
    CLIP_CHECKPOINT: str = "tsbir_model_weight/tsbir_model_final.pt"
    SIGLIP2_MODEL: str = "google/siglip2-base-patch16-512"
    OBJECT_SCORE_THRESHOLD: float = 0.3
    OBJECT_MAX_PER_FRAME: int = 20
    IMAGE_BASE_URL: str = "/keyframes"
    GCP_PROJECT_ID: str = ""
    GOOGLE_APPLICATION_CREDENTIALS: str = ""

    # LLM / VLM (OpenRouter)
    LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = ""
    VLM_MODEL: str = ""
    LLM_SSL_VERIFY: bool = True
    LLM_CA_BUNDLE: str = ""

    # Observability
    TRACES_DB: str = "data/traces.db"
    INDEX_VERSION: str = "v1"

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

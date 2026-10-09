import os
from functools import lru_cache
from pathlib import Path

from .config import settings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    creds = settings.GOOGLE_APPLICATION_CREDENTIALS
    if creds:
        creds_path = Path(creds)
        if not creds_path.is_absolute():
            creds_path = _PROJECT_ROOT / creds_path
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)
        print(f"GCP credentials: {creds_path} (exists={creds_path.exists()})")

    from google.cloud import translate_v2 as translate
    _client = translate.Client()
    return _client


@lru_cache(maxsize=256)
def translate_vi_to_en(text: str) -> str:
    client = _get_client()
    result = client.translate(text, source_language="vi", target_language="en")
    return result["translatedText"]

import html
import os
import ssl
from functools import lru_cache

import httpx

_TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"

_http_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _http_client
    if _http_client is not None:
        return _http_client

    verify: ssl.SSLContext | bool = True
    if os.name == "nt":
        import truststore
        verify = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    _http_client = httpx.Client(verify=verify, timeout=10)
    return _http_client


@lru_cache(maxsize=256)
def translate_vi_to_en(text: str) -> str:
    resp = _get_client().get(
        _TRANSLATE_URL,
        params={"client": "gtx", "sl": "vi", "tl": "en", "dt": "t", "q": text},
    )
    resp.raise_for_status()
    data = resp.json()
    translated = "".join(segment[0] for segment in data[0] if segment[0])
    return html.unescape(translated)

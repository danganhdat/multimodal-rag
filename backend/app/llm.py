"""LLM / VLM factory: one place to create ChatOpenAI models.

Uses OpenRouter (or any OpenAI-compatible endpoint) via langchain-openai.
No API key → raises LLMUnavailable with a clear message.
"""

import base64
import io
import os
import ssl
from contextvars import ContextVar
from pathlib import Path

import httpx

from .config import settings

LLM_CALLBACKS = ContextVar("llm_callbacks", default=None)


class LLMUnavailable(Exception):
    """Raised when LLM/VLM is not configured (no API key)."""
    pass


def get_chat(kind: str = "text", **kwargs):
    """Get a ChatOpenAI instance.

    Args:
        kind: "text" for the LLM controller, "vlm" for the vision model.
        **kwargs: Extra kwargs passed to ChatOpenAI.

    Returns:
        langchain_openai.ChatOpenAI instance.

    Raises:
        LLMUnavailable: if LLM_API_KEY is not set.
    """
    if kwargs.get("callbacks") is None and LLM_CALLBACKS.get():
        kwargs["callbacks"] = LLM_CALLBACKS.get()
    if not settings.LLM_API_KEY:
        raise LLMUnavailable(
            "LLM_API_KEY is not set. Set it in .env to use LLM features. "
            "Get a key at https://openrouter.ai/keys"
        )

    from langchain_openai import ChatOpenAI

    model = settings.VLM_MODEL if kind == "vlm" else settings.LLM_MODEL
    if not model:
        raise LLMUnavailable(
            f"{'VLM_MODEL' if kind == 'vlm' else 'LLM_MODEL'} is not set in .env."
        )

    # Build httpx client with optional CA bundle / SSL verify
    verify = settings.LLM_CA_BUNDLE if settings.LLM_CA_BUNDLE else settings.LLM_SSL_VERIFY
    if verify is True and os.name == "nt":
        import truststore

        verify = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    http_client = httpx.Client(verify=verify)
    http_async_client = httpx.AsyncClient(verify=verify)

    return ChatOpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        model=model,
        timeout=kwargs.pop("timeout", 30),
        max_retries=kwargs.pop("max_retries", 2),
        http_client=http_client,
        http_async_client=http_async_client,
        **kwargs,
    )


def image_message(text: str, image_paths: list[str | Path], max_side: int = 768):
    """Build a HumanMessage with text + base64 images for VLM.

    Args:
        text: The text prompt.
        image_paths: Paths to image files.
        max_side: Max dimension to resize images to (saves tokens).

    Returns:
        langchain_core.messages.HumanMessage with content blocks.
    """
    from langchain_core.messages import HumanMessage
    from PIL import Image

    content = [{"type": "text", "text": text}]

    for img_path in image_paths:
        img = Image.open(str(img_path)).convert("RGB")
        # Resize if too large
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        # Encode to base64 JPEG
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    return HumanMessage(content=content)

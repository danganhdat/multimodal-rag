import asyncio
import logging

from fastapi import APIRouter

from ..models.schemas import TranslateRequest, TranslateResponse

logger = logging.getLogger(__name__)
from ..translator import translate_vi_to_en

router = APIRouter()


@router.post("/translate", response_model=TranslateResponse)
async def translate(body: TranslateRequest):
    text = body.text.strip()
    if not text:
        return TranslateResponse(original=body.text, translated="")

    try:
        translated = await asyncio.to_thread(translate_vi_to_en, text)
    except Exception as e:
        logger.warning("Translation failed: %s", e)
        translated = body.text

    return TranslateResponse(original=body.text, translated=translated)

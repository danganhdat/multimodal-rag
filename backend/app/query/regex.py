"""Regex-based pre-LLM extraction: quoted text → OCR, numbers → OCR, keywords → COCO objects.

Runs instantly before the LLM call. Results are shown in the UI immediately.
"""

import json
import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Small vi→COCO dictionary for common Vietnamese terms
_VI_TO_COCO = {
    "người": "person", "xe hơi": "car", "xe ô tô": "car", "ô tô": "car",
    "xe máy": "motorcycle", "xe đạp": "bicycle", "xe buýt": "bus", "xe bus": "bus",
    "xe tải": "truck", "máy bay": "airplane", "tàu": "boat", "thuyền": "boat",
    "đèn giao thông": "traffic light", "chó": "dog", "mèo": "cat",
    "ghế": "chair", "bàn": "dining table", "ti vi": "tv", "tivi": "tv",
    "điện thoại": "cell phone", "laptop": "laptop", "cốc": "cup",
    "chai": "bottle", "ô": "umbrella", "dù": "umbrella",
    "ba lô": "backpack", "túi xách": "handbag", "vali": "suitcase",
}

_coco_classes: list[str] | None = None


def _load_coco_classes() -> list[str]:
    global _coco_classes
    if _coco_classes is not None:
        return _coco_classes
    obj_path = _PROJECT_ROOT / "object_classes.json"
    if obj_path.exists():
        _coco_classes = json.loads(obj_path.read_text(encoding="utf-8"))
    else:
        _coco_classes = []
    return _coco_classes


def extract_ocr(text: str) -> list[str]:
    """Extract OCR candidates from text.

    - Quoted strings: "HTV9", 'SALE'
    - ALL CAPS words (≥2 chars)
    - Numbers with context: 2025, 50%
    """
    results = []

    # Quoted strings
    for m in re.finditer(r'["“”]([^""“”]+)["“”]', text):
        results.append(m.group(1).strip())
    for m in re.finditer(r"'([^']+)'", text):
        results.append(m.group(1).strip())

    # ALL CAPS words (at least 2 characters, possibly with digits)
    for m in re.finditer(r"\b([A-Z][A-Z0-9]{1,})\b", text):
        word = m.group(1)
        if word not in ("AND", "OR", "THE", "NOT", "FOR", "WITH"):
            results.append(word)

    # Numbers that look like they'd appear on screen
    for m in re.finditer(r"\b(\d{2,}%?)\b", text):
        results.append(m.group(1))

    return list(dict.fromkeys(results))  # dedupe preserving order


def extract_objects(text: str) -> list[str]:
    """Extract COCO object labels from text.

    Checks both English COCO names and Vietnamese→COCO mappings.
    """
    coco = _load_coco_classes()
    text_lower = text.lower()
    found = []

    # Check Vietnamese terms first (longer matches first to avoid partial)
    for vi, en in sorted(_VI_TO_COCO.items(), key=lambda x: -len(x[0])):
        if vi in text_lower and en not in found:
            found.append(en)

    # Check COCO class names in English
    for cls in coco:
        if cls.lower() in text_lower and cls not in found:
            found.append(cls)

    return found


def regex_parse(text: str) -> dict:
    """Run regex extraction on raw question text.

    Returns:
        {"ocr": [...], "objects": [...]}
    """
    return {
        "ocr": extract_ocr(text),
        "objects": extract_objects(text),
    }

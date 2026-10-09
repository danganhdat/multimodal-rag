import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def rerank(
        self, query: str, candidates: list[dict], top_k: int
    ) -> list[dict]:
        pairs = []
        for c in candidates:
            doc = " ".join(c.get("objects", []))
            ocr = c.get("ocr", "").strip()
            if ocr:
                doc = f"{doc} {ocr}" if doc else ocr
            pairs.append([query, doc])
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, padding=True, truncation=True,
                return_tensors="pt", max_length=512,
            )
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float().tolist()
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]

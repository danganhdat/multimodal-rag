"""In-memory BM25 index over Milvus OCR field.

Built at startup from iter_ocr(). Searches lexically: OCR tokens from the
user/QE are matched against OCR stored in Milvus metadata.

Tokenizer: NFC → lowercase → strip accents (đ→d) → \\w+ split.
"""

import math
import re
import unicodedata
from collections import defaultdict

from ..observe import span


def strip_accents(text: str) -> str:
    """Remove Vietnamese diacritics: đ→d, then strip combining marks."""
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def tokenize(text: str) -> list[str]:
    """NFC → lowercase → strip accents → \\w+ tokens."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = strip_accents(text)
    return re.findall(r"\w+", text)


class BM25Index:
    """Simple in-memory BM25 index. k1=1.5, b=0.75."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_lengths: list[int] = []
        self.avgdl: float = 0.0
        self.n_docs: int = 0
        # inverted index: token → {doc_idx: tf}
        self.index: dict[str, dict[int, int]] = defaultdict(dict)
        self.df: dict[str, int] = defaultdict(int)
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def build(self, ocr_iter):
        """Build index from an iterable of (id, ocr_text) pairs."""
        for doc_id, ocr_text in ocr_iter:
            tokens = tokenize(ocr_text)
            if not tokens:
                continue
            idx = len(self.doc_ids)
            self.doc_ids.append(doc_id)
            self.doc_lengths.append(len(tokens))

            seen = set()
            for t in tokens:
                if t not in self.index:
                    self.index[t] = {}
                self.index[t][idx] = self.index[t].get(idx, 0) + 1
                if t not in seen:
                    self.df[t] += 1
                    seen.add(t)

        self.n_docs = len(self.doc_ids)
        self.avgdl = sum(self.doc_lengths) / self.n_docs if self.n_docs else 1.0
        self._ready = self.n_docs > 0
        return self

    def search(self, query: str, top_k: int = 200) -> list[tuple[str, float]]:
        """Search and return [(doc_id, score)] sorted by score desc."""
        query_tokens = tokenize(query)
        if not query_tokens or not self._ready:
            return []

        scores: dict[int, float] = defaultdict(float)
        for qt in query_tokens:
            postings = self.index.get(qt)
            if not postings:
                continue
            n_t = self.df.get(qt, 0)
            idf = math.log((self.n_docs - n_t + 0.5) / (n_t + 0.5) + 1)
            for doc_idx, tf in postings.items():
                dl = self.doc_lengths[doc_idx]
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
                scores[doc_idx] += idf * tf_norm

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.doc_ids[idx], score) for idx, score in ranked]


def search_bm25(
    bm25: BM25Index,
    ocr_queries: list[str],
    top_n: int = 200,
    trace_id: str = "",
) -> list[list[tuple[str, float]]]:
    """Run BM25 search for each OCR query string.

    Returns one result-list per query. Each item is (kf_id, score).
    Skips queries that are empty.
    """
    results = []
    with span(trace_id, "search.bm25", n_queries=len(ocr_queries), ready=bm25.ready) as attrs:
        for q in ocr_queries:
            if q.strip():
                results.append(bm25.search(q, top_k=top_n))
            else:
                results.append([])
        attrs["n_results"] = sum(len(r) for r in results)
    return results

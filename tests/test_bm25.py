"""Unit tests for the BM25 tokenizer and scorer (will be in retrieve/bm25.py).

Tests use a minimal BM25 implementation to validate the algorithm
before the actual module is built.
"""

import math
import re
import unicodedata


def _strip_accents(text: str) -> str:
    """Remove Vietnamese diacritics: NFC → NFD → strip combining marks, đ→d."""
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text


def tokenize(text: str) -> list[str]:
    """NFC → lowercase → strip accents → \\w+ tokens."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = _strip_accents(text)
    return re.findall(r"\w+", text)


def bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    doc_lengths: list[int],
    df: dict[str, int],
    n_docs: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """BM25 score for one document."""
    avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1
    dl = len(doc_tokens)
    tf_map: dict[str, int] = {}
    for t in doc_tokens:
        tf_map[t] = tf_map.get(t, 0) + 1

    score = 0.0
    for qt in query_tokens:
        tf = tf_map.get(qt, 0)
        if tf == 0:
            continue
        n_t = df.get(qt, 0)
        idf = math.log((n_docs - n_t + 0.5) / (n_t + 0.5) + 1)
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
        score += idf * tf_norm
    return score


# ── Tests ──

def test_tokenize_basic():
    assert tokenize("Hello World") == ["hello", "world"]


def test_tokenize_vietnamese():
    tokens = tokenize("Đà Nẵng")
    assert tokens == ["da", "nang"]


def test_tokenize_numbers():
    assert tokenize("HTV9 Channel 2025") == ["htv9", "channel", "2025"]


def test_tokenize_accents_stripped():
    """Florence OCR often drops accents; our tokenizer should match either way."""
    assert tokenize("Hà Nội") == tokenize("Ha Noi")


def test_bm25_exact_match():
    query = tokenize("hello world")
    doc = tokenize("hello world program")
    score = bm25_score(query, doc, [3], {"hello": 1, "world": 1}, n_docs=1)
    assert score > 0


def test_bm25_no_match():
    query = tokenize("hello")
    doc = tokenize("goodbye")
    score = bm25_score(query, doc, [1], {"goodbye": 1}, n_docs=1)
    assert score == 0.0


def test_bm25_tf_matters():
    """Document with more occurrences of query term should score higher."""
    query = tokenize("htv9")
    doc1 = tokenize("htv9 news")
    doc2 = tokenize("htv9 htv9 htv9 news")
    lengths = [2, 4]
    df = {"htv9": 2, "news": 2}
    s1 = bm25_score(query, doc1, lengths, df, n_docs=2)
    s2 = bm25_score(query, doc2, lengths, df, n_docs=2)
    assert s2 > s1


def test_bm25_idf_matters():
    """Rare terms should have higher IDF and thus higher score contribution."""
    query = tokenize("rare")
    doc = tokenize("rare common")
    # rare appears in 1/10 docs, common in 9/10
    df_rare = {"rare": 1, "common": 9}
    score_rare = bm25_score(tokenize("rare"), doc, [2], df_rare, n_docs=10)
    score_common = bm25_score(tokenize("common"), doc, [2], df_rare, n_docs=10)
    assert score_rare > score_common


def test_bm25_empty_query():
    score = bm25_score([], tokenize("hello world"), [2], {"hello": 1}, n_docs=1)
    assert score == 0.0

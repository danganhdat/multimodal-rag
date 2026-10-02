from FlagEmbedding import FlagReranker


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = FlagReranker(model_name, use_fp16=True)

    def rerank(
        self, query: str, candidates: list[dict], top_k: int
    ) -> list[dict]:
        pairs = [[query, " ".join(c.get("objects", []))] for c in candidates]
        scores = self.model.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]

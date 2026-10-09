from pymilvus import (
    AnnSearchRequest,
    MilvusClient,
    RRFRanker,
    WeightedRanker,
)


class MilvusSearchService:
    def __init__(self, uri: str, collection_name: str):
        self.client = MilvusClient(uri=uri)
        self.collection = collection_name

    OUTPUT_FIELDS = [
        "video_name",
        "keyframe_idx",
        "frame_idx",
        "pts_time",
        "fps",
        "objects",
        "ocr",
    ]

    def search_single(
        self,
        vector: list[float],
        top_k: int = 50,
        metric_type: str = "COSINE",
        filter_expr: str | None = None,
        group_by_video: bool = False,
        anns_field: str = "clip_vector",
    ) -> list[dict]:
        params = {"metric_type": metric_type, "params": {"ef": 512}}
        kwargs: dict = {
            "collection_name": self.collection,
            "data": [vector],
            "anns_field": anns_field,
            "limit": top_k,
            "search_params": params,
            "output_fields": self.OUTPUT_FIELDS,
        }
        if filter_expr:
            kwargs["filter"] = filter_expr
        if group_by_video:
            kwargs["group_by_field"] = "video_name"

        results = self.client.search(**kwargs)
        return self._flatten(results)

    def search_hybrid(
        self,
        vectors: list[list[float]],
        top_k: int = 50,
        sub_k: int = 100,
        metric_type: str = "COSINE",
        ranker_type: str = "rrf",
        rrf_k: int = 60,
        weights: list[float] | None = None,
        filter_expr: str | None = None,
        anns_fields: list[str] | None = None,
    ) -> list[dict]:
        reqs = []
        for i, vec in enumerate(vectors):
            field = (anns_fields[i] if anns_fields and i < len(anns_fields)
                     else "clip_vector")
            req = AnnSearchRequest(
                data=[vec],
                anns_field=field,
                param={"metric_type": metric_type, "params": {"ef": 512}},
                limit=sub_k,
                expr=filter_expr or "",
            )
            reqs.append(req)

        if ranker_type == "weighted" and weights:
            ranker = WeightedRanker(*weights)
        else:
            ranker = RRFRanker(k=rrf_k)

        results = self.client.hybrid_search(
            collection_name=self.collection,
            reqs=reqs,
            ranker=ranker,
            limit=top_k,
            output_fields=self.OUTPUT_FIELDS,
        )
        return self._flatten(results)

    def get_neighbors(
        self,
        video_name: str,
        keyframe_idx: int,
        window: int = 5,
    ) -> list[dict]:
        low = max(1, keyframe_idx - window)
        high = keyframe_idx + window
        expr = (
            f'video_name == "{video_name}" '
            f"and keyframe_idx >= {low} "
            f"and keyframe_idx <= {high}"
        )
        results = self.client.query(
            collection_name=self.collection,
            filter=expr,
            output_fields=["keyframe_idx", "frame_idx", "pts_time"],
        )
        return sorted(results, key=lambda r: r["keyframe_idx"])

    @staticmethod
    def _flatten(results: list) -> list[dict]:
        hits = []
        for result_set in results:
            for hit in result_set:
                entry = {
                    "id": hit["id"],
                    "score": hit["distance"],
                }
                entity = hit.get("entity", {})
                entry.update(entity)
                hits.append(entry)
        return hits

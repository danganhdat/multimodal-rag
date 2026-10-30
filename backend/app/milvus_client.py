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
        anns_field: str = "text_vector",
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

    def search_batch(
        self,
        vectors: list[list[float]],
        anns_field: str = "text_vector",
        top_k: int = 200,
        metric_type: str = "COSINE",
        filter_expr: str | None = None,
    ) -> list[list[dict]]:
        """Search multiple vectors in one call (nq = len(vectors)).

        Returns a list of hit-lists, one per input vector.
        """
        params = {"metric_type": metric_type, "params": {"ef": 512}}
        kwargs: dict = {
            "collection_name": self.collection,
            "data": vectors,
            "anns_field": anns_field,
            "limit": top_k,
            "search_params": params,
            "output_fields": self.OUTPUT_FIELDS,
        }
        if filter_expr:
            kwargs["filter"] = filter_expr

        results = self.client.search(**kwargs)
        # results is list[list[hit]], one inner list per query vector
        return [
            [{"id": h["id"], "score": h["distance"], **h.get("entity", {})} for h in result_set]
            for result_set in results
        ]

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
                     else "text_vector")
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

    def get_neighbors_batch(
        self,
        items: list[tuple[str, int]],
        window: int = 5,
    ) -> dict[str, list[dict]]:
        """Batch neighbor lookup. items = [(video_name, keyframe_idx), ...].

        Returns {id: [neighbor_dicts]} keyed by "video/nnn".
        """
        if not items:
            return {}

        # Group by video to reduce queries
        by_video: dict[str, list[int]] = {}
        for vname, kidx in items:
            by_video.setdefault(vname, []).append(kidx)

        all_neighbors: dict[str, list[dict]] = {}
        for vname, kidxs in by_video.items():
            lo = max(1, min(kidxs) - window)
            hi = max(kidxs) + window
            expr = (
                f'video_name == "{vname}" '
                f"and keyframe_idx >= {lo} "
                f"and keyframe_idx <= {hi}"
            )
            rows = self.client.query(
                collection_name=self.collection,
                filter=expr,
                output_fields=["keyframe_idx", "frame_idx", "pts_time"],
            )
            rows.sort(key=lambda r: r["keyframe_idx"])
            for kidx in kidxs:
                key = f"{vname}/{kidx:03d}"
                all_neighbors[key] = [
                    r for r in rows
                    if abs(r["keyframe_idx"] - kidx) <= window and r["keyframe_idx"] != kidx
                ]
        return all_neighbors

    def get_by_ids_expr(
        self,
        filter_expr: str,
        output_fields: list[str] | None = None,
        limit: int = 10000,
    ) -> list[dict]:
        """Fetch rows matching a filter expression."""
        fields = output_fields or self.OUTPUT_FIELDS
        return self.client.query(
            collection_name=self.collection,
            filter=filter_expr,
            output_fields=fields,
            limit=limit,
        )

    def get_by_ids(self, ids: list[str], output_fields: list[str] | None = None) -> list[dict]:
        """Fetch rows by primary key."""
        if not ids:
            return []
        fields = output_fields or self.OUTPUT_FIELDS
        return self.client.get(
            collection_name=self.collection,
            ids=ids,
            output_fields=fields,
        )

    def get_vectors(self, ids: list[str], field: str = "text_vector") -> dict[str, list[float]]:
        """Fetch vector values for given ids. Returns {id: vector}."""
        if not ids:
            return {}
        rows = self.client.get(
            collection_name=self.collection,
            ids=ids,
            output_fields=[field],
        )
        return {r["id"]: r[field] for r in rows}

    def iter_ocr(self, batch_size: int = 5000):
        """Iterate all (id, ocr) pairs using query_iterator. For BM25 index build."""
        it = self.client.query_iterator(
            collection_name=self.collection,
            output_fields=["id", "ocr"],
            batch_size=batch_size,
        )
        while True:
            rows = it.next()
            if not rows:
                break
            for r in rows:
                ocr = r.get("ocr", "")
                if ocr:
                    yield r["id"], ocr
        it.close()

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

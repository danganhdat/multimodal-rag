"""Dense ANN search on Milvus text_vector field using SigLIP2."""

from ..milvus_client import MilvusSearchService
from ..observe import span


def search_dense(
    milvus: MilvusSearchService,
    vectors: list[list[float]],
    top_n: int = 200,
    filter_expr: str | None = None,
    trace_id: str = "",
) -> list[list[dict]]:
    """Run dense ANN search for each query vector.

    Returns one hit-list per vector. Each hit has: id, score, video_name,
    keyframe_idx, frame_idx, pts_time, fps, objects, ocr.
    """
    with span(trace_id, "search.dense", n_vectors=len(vectors), top_n=top_n):
        return milvus.search_batch(
            vectors=vectors,
            anns_field="text_vector",
            top_k=top_n,
            filter_expr=filter_expr,
        )

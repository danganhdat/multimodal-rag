"""v1 feedback endpoint: /api/v1/feedback."""

from fastapi import APIRouter
from pydantic import BaseModel

from ...observe import write_feedback

router = APIRouter()


class FeedbackRequest(BaseModel):
    trace_id: str
    rating: int = 0
    note: str = ""
    kf_ids: list[str] = []
    client_total_ms: float = 0


@router.post("/feedback")
async def v1_feedback(body: FeedbackRequest):
    write_feedback(
        trace_id=body.trace_id,
        rating=body.rating,
        note=body.note,
        kf_ids=body.kf_ids or None,
        client_total_ms=body.client_total_ms,
    )
    return {"ok": True}

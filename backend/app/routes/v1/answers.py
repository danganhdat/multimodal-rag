"""v1 answers export endpoint: /api/v1/answers/export."""

from fastapi import APIRouter, Query

from ...observe import get_answers

router = APIRouter()


@router.get("/answers/export")
async def v1_answers_export(
    session_id: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    """Export answers as JSON for evaluation or review."""
    return {"answers": get_answers(session_id=session_id, limit=limit)}

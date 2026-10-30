"""v1 query endpoints: /api/v1/query/parse."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ...config_flags import load_flags
from ...observe import span, write_trace
from ...query.qe import parse_query, ParsedQuery
from ...query.regex import regex_parse

router = APIRouter()


class ParseRequest(BaseModel):
    question: str
    lang: str = "auto"
    flags: dict | None = None


class ParseResponse(BaseModel):
    parsed: ParsedQuery
    regex: dict
    trace_id: str = ""


@router.post("/query/parse", response_model=ParseResponse)
async def v1_parse(body: ParseRequest, request: Request):
    """Parse a question into structured search events.

    Returns both the LLM-parsed result and the instant regex extraction.
    If LLM is unavailable, parsed.fallback=true and a warning is shown.
    """
    trace_id = getattr(request.state, "trace_id", "")
    flags = load_flags(body.flags)

    # Regex runs instantly (no LLM needed)
    regex_result = regex_parse(body.question)

    # LLM parse (may fall back)
    parsed = await parse_query(
        question=body.question,
        lang=body.lang,
        flags=flags,
        trace_id=trace_id,
    )

    write_trace(
        trace_id,
        endpoint="/api/v1/query/parse",
        mode="parse",
        query_raw=body.question,
    )

    return ParseResponse(
        parsed=parsed,
        regex=regex_result,
        trace_id=trace_id,
    )

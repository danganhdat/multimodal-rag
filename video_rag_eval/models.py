"""Validated contracts for every external model response."""

from pydantic import BaseModel, Field, field_validator, model_validator


class VisualFact(BaseModel):
    fact_id: str
    type: str
    claim: str
    transcription: str | None = None
    confidence: float | None = None
    bbox: list[float] | None = None

    @field_validator("fact_id", mode="before")
    @classmethod
    def normalize_fact_id(cls, value):
        return str(value)


class FactResponse(BaseModel):
    visual_facts: list[VisualFact]
    uncertainties: list[str] = Field(default_factory=list)
    usable_question_types: list[str] = Field(default_factory=list)


class QAResponse(BaseModel):
    description_vi: str
    question_vi: str
    reference_answer: str
    retrieval_query_en: str
    question_type: str
    required_claims: list[str]
    answerable: bool


class VerifyResponse(BaseModel):
    independent_answer: str
    answerable: bool
    description_grounded: bool
    question_unambiguous: bool
    leaks_answer: bool
    needs_more_pixels: bool
    reasons: list[str]


class EquivalenceResponse(BaseModel):
    equivalent: bool
    reason: str


class QrelResponse(BaseModel):
    relevance: int = Field(ge=0, le=3)
    derived_answer: str
    uncertain: bool
    reason_code: str
    supports_claim_ids: list[int] = Field(default_factory=list)


class QueryResponse(BaseModel):
    queries_en: list[str] = Field(min_length=1)
    objects: list[str] = Field(default_factory=list)


class CropResponse(BaseModel):
    bbox: list[float] | None = None


class SummaryResponse(BaseModel):
    summary: str


class AnswerResponse(BaseModel):
    status: str
    answer: str
    citations: list[str]


class SupportResponse(BaseModel):
    supported: list[bool]


class ClaimsResponse(BaseModel):
    claims: list[str] = Field(min_length=1, max_length=6)


class FaithfulnessResponse(BaseModel):
    claims: list[str]
    claim_support: list[bool]

    @model_validator(mode="after")
    def aligned_claims(self):
        if len(self.claims) != len(self.claim_support):
            raise ValueError("faithfulness claims and verdicts must have equal length")
        return self

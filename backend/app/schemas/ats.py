from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class ATSAnalysisRequest(BaseModel):
    resume_text: str = Field(..., min_length=1)
    job_text: str = Field(..., min_length=1)


class DocumentParseRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source_name: str = "inline_text"


class JDStructureRequest(BaseModel):
    job_text: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Shared sub-schemas
# ---------------------------------------------------------------------------

class InterviewFocusResponse(BaseModel):
    must_probe: list[str] = Field(default_factory=list)
    strengths_to_confirm: list[str] = Field(default_factory=list)
    suggested_question_themes: list[str] = Field(default_factory=list)
    recommended_depth: str = "standard"


class CandidateEvaluationResponse(BaseModel):
    score: float = 0.0
    grade: str = "E"
    overall_summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    interview_focus: InterviewFocusResponse = Field(default_factory=InterviewFocusResponse)


class StructuredJobDescriptionResponse(BaseModel):
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    experience_years: str | None = None
    education_level: str | None = None
    responsibilities: list[str] = Field(default_factory=list)

    @field_validator("required_skills", "preferred_skills", "responsibilities", mode="before")
    @classmethod
    def default_null_lists(cls, value: Any) -> list[str] | Any:
        return [] if value is None else value


# ---------------------------------------------------------------------------
# Score / breakdown responses (no LLM)
# ---------------------------------------------------------------------------

class ScoreResponse(BaseModel):
    score: float
    grade: str


class BreakdownResponse(BaseModel):
    score: float
    grade: str
    semantic_similarity: float
    keyword_coverage: float
    section_completeness: float
    section_scores: dict[str, float]


class KeywordsResponse(BaseModel):
    keyword_coverage: float
    missing_keywords: list[str]
    keyword_matches: list[dict]


# ---------------------------------------------------------------------------
# LLM-enriched responses
# ---------------------------------------------------------------------------

class EvaluationResponse(BaseModel):
    """Full evaluation response returned by /ats/evaluate and /ats/evaluate-files."""
    score: float
    grade: str
    semantic_similarity: float
    keyword_coverage: float
    section_completeness: float
    section_scores: dict[str, float]
    missing_keywords: list[str]
    keyword_matches: list[dict]
    structured_job_description: dict | None = None
    evaluation: CandidateEvaluationResponse | None = None
    # Legacy field - kept for backward compat
    suggestions: list[str] = Field(default_factory=list)


class LLMAnalysisResponse(BaseModel):
    structured_job_description: dict | None = None
    evaluation: CandidateEvaluationResponse | None = None
    # Legacy field
    suggestions: list[str] = Field(default_factory=list)


# Legacy aliases used by existing route imports.
ATSResultResponse = EvaluationResponse


class SuggestionsResponse(BaseModel):
    score: float
    grade: str
    evaluation: CandidateEvaluationResponse | None = None
    suggestions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing responses
# ---------------------------------------------------------------------------

class ParsedDocumentResponse(BaseModel):
    source_path: str
    raw_text: str
    sections: dict[str, str]
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class ParsedPairResponse(BaseModel):
    resume: ParsedDocumentResponse
    job: ParsedDocumentResponse

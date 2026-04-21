from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ExtractedKeyword:
    phrase: str
    score: float


@dataclass(slots=True)
class KeywordMatch:
    job_keyword: str
    resume_keyword: str
    similarity: float


@dataclass(slots=True)
class ParsedDocument:
    source_path: str
    raw_text: str
    sections: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StructuredJobDescription:
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    experience_years: str | None = None
    education_level: str | None = None
    responsibilities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InterviewFocus:
    must_probe: list[str] = field(default_factory=list)
    strengths_to_confirm: list[str] = field(default_factory=list)
    suggested_question_themes: list[str] = field(default_factory=list)
    recommended_depth: str = "standard"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CandidateEvaluation:
    score: float = 0.0
    grade: str = "E"
    overall_summary: str = ""
    strengths: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    interview_focus: InterviewFocus = field(default_factory=InterviewFocus)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LLMCandidateAssessment:
    structured_resume: dict[str, Any] = field(default_factory=dict)
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    inferred_relevant_skills: list[str] = field(default_factory=list)
    grade: str = "E"
    hiring_potential: str = "low"
    hire_recommendation: str = "not_recommended"
    rationale: str = ""
    strengths: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    interview_focus_areas: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ATSResult:
    score: float
    grade: str
    semantic_similarity: float
    keyword_coverage: float
    section_completeness: float
    section_scores: dict[str, float]
    missing_keywords: list[str]
    keyword_matches: list[KeywordMatch]
    structured_job_description: StructuredJobDescription | None = None
    evaluation: CandidateEvaluation | None = None
    llm_assessment: LLMCandidateAssessment | None = None
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.structured_job_description is not None:
            payload["structured_job_description"] = self.structured_job_description.to_dict()
        if self.evaluation is not None:
            payload["evaluation"] = self.evaluation.to_dict()
        if self.llm_assessment is not None:
            payload["llm_assessment"] = self.llm_assessment.to_dict()
        return payload

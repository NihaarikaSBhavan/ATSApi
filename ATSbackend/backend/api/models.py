"""Pydantic models for ATS API."""
from pydantic import BaseModel
from typing import Dict, List


class ATSResult(BaseModel):
    """Result of ATS resume analysis."""
    score: float
    label: str
    resume_skills: List[str]
    jd_skills: List[str]

    # Exact string intersection (backward compatible)
    matched_skills: List[str]

    # Semantic soft matches: JD skill -> [best_resume_skill, cosine_sim]
    soft_matched_skills: Dict = {}

    # True gap: JD skills with neither exact nor semantic match
    missing_skills: List[str]

    # Fraction of JD skills covered (exact + soft), in [0, 1]
    skill_coverage: float = 0.0

    features: dict
    inferred_skills: List[str]
    skill_gap: List[dict]
    inferred_skill_confidences: Dict = {}
    domain: str = "generic"
    explanation: str = ""

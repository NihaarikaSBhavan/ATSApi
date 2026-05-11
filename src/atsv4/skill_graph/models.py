from __future__ import annotations
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


# Used only as an LLM prompt hint in extractor.py — not enforced on the model.
# The graph is domain-agnostic; any string is a valid domain.
DOMAIN_HINTS = [
    "Software Engineering",
    "Data Science / ML",
    "Design / UX",
    "Product Management",
    "Marketing / Growth",
    "Finance / Accounting",
    "Healthcare / Medical",
    "DevOps / Infra",
    "Cybersecurity",
    "Legal / Compliance",
    "Sales / BD",
    "Operations / PM",
    "Other",
]

# These are genuinely closed enumerations — every valid value is known upfront.
SKILL_LEVELS = Literal["beginner", "intermediate", "advanced", "expert"]

CATEGORIES = Literal[
    "language",
    "framework",
    "tool",
    "methodology",
    "soft skill",
    "certification",
    "concept",
    "platform",
]

RELATION_TYPES = Literal[
    "requires",
    "related to",
    "subset of",
    "leads to",
    "alternative to",
    "enhances",
]

SOURCE_TYPES = Literal["job_description", "resume", "linkedin", "manual"]


class Skill(BaseModel):
    id: Optional[str] = None          # UUID set by DB
    name: str
    domain: str                        # open — any domain the LLM or user supplies
    level: SKILL_LEVELS = "intermediate"
    category: CATEGORIES = "tool"
    source: SOURCE_TYPES = "manual"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class SkillEdge(BaseModel):
    id: Optional[str] = None
    source_id: str
    target_id: str
    relation: RELATION_TYPES
    created_at: Optional[datetime] = None


class RawSkill(BaseModel):
    """Intermediate model returned by LLM before DB merge."""
    name: str
    domain: str
    level: str = "intermediate"
    category: str = "tool"
    relations: list[dict] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    detected_domain: str
    secondary_domains: list[str] = Field(default_factory=list)
    summary: str = ""
    raw_skills: list[RawSkill] = Field(default_factory=list)
    new_skills_added: int = 0
    new_edges_added: int = 0
    total_skills_in_graph: int = 0
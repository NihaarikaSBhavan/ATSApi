from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(slots=True)
class ATSConfig:
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv("ATS_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    semantic_weight: float = 0.45
    keyword_weight: float = 0.35
    completeness_weight: float = 0.20
    semantic_section_weights: dict[str, float] = field(
        default_factory=lambda: {
            "summary": 0.15,
            "experience": 0.45,
            "skills": 0.30,
            "education": 0.10,
        }
    )
    keyphrase_ngram_range: tuple[int, int] = (1, 2)
    jd_top_keywords: int = 20
    resume_top_keywords: int = 30
    fuzzy_match_threshold: float = 84.0
    use_mmr: bool = True
    diversity: float = 0.5
    llm_model_name: str = field(
        default_factory=lambda: os.getenv("AI_MODEL")
        or os.getenv("ATS_LLM_MODEL")
        or "gpt-4o-mini"
    )
    llm_api_key: str | None = field(
        default_factory=lambda: os.getenv("LLM_API_KEY")
        or os.getenv("GROK_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    llm_base_url: str | None = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    )

    def validate(self) -> None:
        total_weight = self.semantic_weight + self.keyword_weight + self.completeness_weight
        if round(total_weight, 6) != 1.0:
            raise ValueError(f"Score weights must sum to 1.0, got {total_weight}")

        section_weight_total = sum(self.semantic_section_weights.values())
        if round(section_weight_total, 6) != 1.0:
            raise ValueError(
                f"Semantic section weights must sum to 1.0, got {section_weight_total}"
            )

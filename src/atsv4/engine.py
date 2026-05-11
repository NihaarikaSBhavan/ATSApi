from __future__ import annotations

from pathlib import Path

from .config import ATSConfig
from .grading import to_grade, to_percentage
from .skill_graph_matcher import extract_and_match as skill_graph_extract_and_match
from .llm import evaluate_candidate, structure_job_description
from .models import ATSResult
from .parsers import parse_document, parse_text_document
from .sectioning import section_completeness_score
from .semantic import score_semantic_similarity


class ATSEngine:
    def __init__(self, config: ATSConfig | None = None) -> None:
        self.config = config or ATSConfig()
        self.config.validate()

    def evaluate(
        self,
        resume_path: str | Path,
        job_path: str | Path,
        *,
        use_llm: bool = True,
    ) -> ATSResult:
        resume_doc = parse_document(resume_path)
        job_doc = parse_document(job_path)
        return self._evaluate_documents(
            resume_doc=resume_doc,
            job_doc=job_doc,
            use_llm=use_llm,
        )

    def evaluate_texts(
        self,
        resume_text: str,
        job_text: str,
        *,
        use_llm: bool = True,
    ) -> ATSResult:
        resume_doc = parse_text_document(resume_text, source_name="resume_text")
        job_doc = parse_text_document(job_text, source_name="job_text")
        return self._evaluate_documents(
            resume_doc=resume_doc,
            job_doc=job_doc,
            use_llm=use_llm,
        )

    def _evaluate_documents(
        self,
        *,
        resume_doc,
        job_doc,
        use_llm: bool,
    ) -> ATSResult:
        structured_jd = structure_job_description(job_doc.raw_text, self.config) if use_llm else None

        section_scores, semantic_similarity = score_semantic_similarity(
            resume_doc.sections,
            job_doc.raw_text,
            self.config,
        )

        # ── Skill extraction & matching (skill_graph pipeline) ──────────────
        # Groq-powered two-stage extraction → in-memory graph construction →
        # exact / fuzzy / traversal matching → level-weighted coverage.
        # Falls back to KeyBERT + rapidfuzz automatically when no Groq key.
        (
            keyword_matches,
            missing_keywords,
            keyword_coverage,
            job_keywords,
            resume_keywords,
            graph_context,
        ) = skill_graph_extract_and_match(
            job_doc.raw_text,
            resume_doc.raw_text,
            self.config,
        )

        completeness = section_completeness_score(resume_doc.sections)

        weighted_score = (
            self.config.semantic_weight * semantic_similarity
            + self.config.keyword_weight * keyword_coverage
            + self.config.completeness_weight * completeness
        )
        score = to_percentage(weighted_score)
        result = ATSResult(
            score=score,
            grade=to_grade(score),
            semantic_similarity=round(semantic_similarity, 4),
            keyword_coverage=round(keyword_coverage, 4),
            section_completeness=round(completeness, 4),
            section_scores={name: round(value, 4) for name, value in section_scores.items()},
            missing_keywords=missing_keywords,
            keyword_matches=keyword_matches,
            structured_job_description=structured_jd,
        )

        if use_llm:
            evaluation = evaluate_candidate(
                job_doc.raw_text,
                resume_doc.sections,
                result,
                self.config,
                graph_context=graph_context,
            )
            result.score = evaluation.score
            result.grade = evaluation.grade
            result.evaluation = evaluation
            result.suggestions = (
                [f"[strength] {s}" for s in evaluation.strengths]
                + [f"[risk] {r}" for r in evaluation.risks]
            )

        return result

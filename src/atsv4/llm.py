from __future__ import annotations

import json
from typing import Any

from .config import ATSConfig
from .grading import to_grade, to_percentage
from .models import (
    ATSResult,
    CandidateEvaluation,
    InterviewFocus,
    StructuredJobDescription,
)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_STRUCTURE_JD_PROMPT = (
    "You are a job-description parser. "
    "Return strict JSON only — no prose, no markdown fences. "
    "Extract the job description into an object with exactly these keys: "
    "required_skills (array of strings), "
    "preferred_skills (array of strings), "
    "experience_years (string or null), "
    "education_level (string or null), "
    "responsibilities (array of strings). "
    "Use null for any field that cannot be determined."
)

_EVALUATE_CANDIDATE_PROMPT = """\
You are a senior technical recruiter evaluating a candidate for a specific role.
You will receive the candidate's structured resume, the structured job description, and a \
deterministic evidence report produced by a local scoring engine.

Your job is to score and grade the candidate for this exact role, then produce a \
recruiter-facing evaluation and interview handoff brief.
Return strict JSON only — no prose, no markdown fences.

## Evaluation rules

SCORE & GRADE
- You own the final candidate score and grade.
- Return score as a number from 0 to 100.
- Return grade as one of "A", "B", "C", "D", or "E".
- Grade calibration:
  A = 85-100: excellent fit, most required skills are clearly evidenced
  B = 70-84: strong fit, minor gaps or weaker evidence
  C = 55-69: plausible fit, several gaps or unclear depth
  D = 40-54: weak fit, major gaps against required criteria
  E = 0-39: poor fit or insufficient relevant evidence
- Use the deterministic semantic, keyword, and section signals as evidence, not as the final answer.
- Penalize missing required skills, unclear seniority, unsupported claims, and resumes that do not evidence the JD's core responsibilities.

OVERALL SUMMARY
- 2-3 sentences. State the candidate's fit signal plainly.
- Reference the role title and the single most important strength or gap.
- Avoid hedging phrases like "could potentially" or "may be suitable".

STRENGTHS
- Concrete, evidence-backed positives drawn from the resume.
- Each item must cite a specific skill, project, or experience — not a generic trait.
- 3-5 items maximum.

RISKS
- Genuine gaps, unverified claims, or hiring concerns relative to this JD.
- Distinguish between hard blockers (missing required skill) and soft risks (inferred, not stated).
- 2-4 items maximum. Omit the array if there are genuinely none.

INTERVIEW MODULE HANDOFF
The candidate will next enter an AI interview module in the hiring portal. \
Populate interview_focus to configure that session:

  must_probe — gaps or risks the AI interviewer must validate before the candidate can progress.
               These become mandatory question topics. Phrase as topics, not questions.
               Example: "depth of Kubernetes production experience"

  strengths_to_confirm — claimed strengths the AI should verify with behavioural or technical
               probing. Example: "led a team of 8 engineers — confirm scope and outcomes"

  suggested_question_themes — broader topic areas the AI interviewer should cover given this role.
               These populate optional question pools.
               Example: "system design under load", "conflict resolution in cross-functional teams"

  recommended_depth — one of "surface" (screening), "standard" (full panel), "deep" \
(senior/technical).
               Base this on role seniority and the number of must_probe gaps identified.

## Output schema (return exactly this structure, no extra keys)

{
  "score": 0,
  "grade": "A|B|C|D|E",
  "overall_summary": "string",
  "strengths": ["string"],
  "risks": ["string"],
  "interview_focus": {
    "must_probe": ["string"],
    "strengths_to_confirm": ["string"],
    "suggested_question_themes": ["string"],
    "recommended_depth": "surface|standard|deep"
  }
}\
"""


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def structure_job_description(job_text: str, config: ATSConfig) -> StructuredJobDescription:
    client = _build_openai_client(config)
    response = client.responses.create(
        model=config.llm_model_name,
        input=[
            {"role": "system", "content": _STRUCTURE_JD_PROMPT},
            {"role": "user", "content": job_text},
        ],
    )
    payload = _extract_json_payload(response.output_text)
    return StructuredJobDescription(
        required_skills=payload.get("required_skills", []),
        preferred_skills=payload.get("preferred_skills", []),
        experience_years=payload.get("experience_years"),
        education_level=payload.get("education_level"),
        responsibilities=payload.get("responsibilities", []),
    )


def evaluate_candidate(
    job_text: str,
    resume_sections: dict[str, str],
    result: ATSResult,
    config: ATSConfig,
) -> CandidateEvaluation:
    """
    LLM call #2 — recruiter-facing evaluation + AI interview handoff brief.

    The deterministic context (score, grade, keyword gaps, structured JD) is injected
    into the user message so the model reasons against real numbers without being asked
    to produce them itself.
    """
    client = _build_openai_client(config)

    deterministic_context = {
        "local_score": result.score,
        "local_grade": result.grade,
        "section_scores": result.section_scores,
        "semantic_similarity_pct": round(result.semantic_similarity * 100, 1),
        "keyword_coverage_pct": round(result.keyword_coverage * 100, 1),
        "section_completeness_pct": round(result.section_completeness * 100, 1),
        "missing_keywords": result.missing_keywords,
        "structured_job_description": (
            result.structured_job_description.to_dict()
            if result.structured_job_description is not None
            else None
        ),
    }

    user_payload = json.dumps(
        {
            "job_description": job_text,
            "resume_sections": resume_sections,
            "ats_analysis": deterministic_context,
        },
        indent=2,
    )

    response = client.responses.create(
        model=config.llm_model_name,
        input=[
            {"role": "system", "content": _EVALUATE_CANDIDATE_PROMPT},
            {"role": "user", "content": user_payload},
        ],
    )

    payload = _extract_json_payload(response.output_text)
    focus_raw = payload.get("interview_focus", {})
    score = _normalize_llm_score(payload.get("score"), fallback=result.score)
    grade = _normalize_llm_grade(payload.get("grade"), score)

    return CandidateEvaluation(
        score=score,
        grade=grade,
        overall_summary=payload.get("overall_summary", ""),
        strengths=payload.get("strengths", []),
        risks=payload.get("risks", []),
        interview_focus=InterviewFocus(
            must_probe=focus_raw.get("must_probe", []),
            strengths_to_confirm=focus_raw.get("strengths_to_confirm", []),
            suggested_question_themes=focus_raw.get("suggested_question_themes", []),
            recommended_depth=focus_raw.get("recommended_depth", "standard"),
        ),
    )


# Legacy alias — existing callers of generate_feedback() keep working.
# Flattens strengths + risks into the old list[str] shape with prefixes.
def generate_feedback(
    job_text: str,
    resume_sections: dict[str, str],
    result: ATSResult,
    config: ATSConfig,
) -> list[str]:
    evaluation = evaluate_candidate(job_text, resume_sections, result, config)
    suggestions: list[str] = []
    for s in evaluation.strengths:
        suggestions.append(f"[strength] {s}")
    for r in evaluation.risks:
        suggestions.append(f"[risk] {r}")
    return suggestions


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_openai_client(config: ATSConfig) -> Any:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openai is required for LLM features") from exc

    if not config.llm_api_key:
        raise RuntimeError(
            "Missing LLM API key. Set LLM_API_KEY or OPENAI_API_KEY in your environment."
        )

    client_kwargs: dict[str, Any] = {"api_key": config.llm_api_key}
    if config.llm_base_url:
        client_kwargs["base_url"] = config.llm_base_url

    return OpenAI(**client_kwargs)


def _extract_json_payload(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        lines = text.splitlines()
        if lines and lines[0].lower() == "json":
            lines = lines[1:]
        text = "\n".join(lines)
    return json.loads(text)


def _normalize_llm_score(value: Any, fallback: float = 0.0) -> float:
    try:
        return to_percentage(float(value) / 100)
    except (TypeError, ValueError):
        return fallback


def _normalize_llm_grade(value: Any, score: float) -> str:
    grade = str(value or "").strip().upper()
    return grade if grade in {"A", "B", "C", "D", "E"} else to_grade(score)

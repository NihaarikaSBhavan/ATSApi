from __future__ import annotations

from rapidfuzz import fuzz

from .config import ATSConfig
from .models import ExtractedKeyword, KeywordMatch


def extract_keywords(text: str, config: ATSConfig, top_n: int) -> list[ExtractedKeyword]:
    try:
        from keybert import KeyBERT  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("keybert is required for keyword extraction") from exc

    kw_model = KeyBERT(config.embedding_model_name)
    results = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=config.keyphrase_ngram_range,
        use_mmr=config.use_mmr,
        diversity=config.diversity,
        top_n=top_n,
    )
    return [ExtractedKeyword(phrase=phrase, score=float(score)) for phrase, score in results]


def fuzzy_match_keywords(
    job_keywords: list[ExtractedKeyword],
    resume_keywords: list[ExtractedKeyword],
    threshold: float,
) -> tuple[list[KeywordMatch], list[str], float]:
    matches: list[KeywordMatch] = []
    missing: list[str] = []

    resume_phrases = [item.phrase for item in resume_keywords]
    for job_keyword in job_keywords:
        best_match = ""
        best_score = 0.0
        for resume_phrase in resume_phrases:
            score = float(fuzz.token_set_ratio(job_keyword.phrase, resume_phrase))
            if score > best_score:
                best_match = resume_phrase
                best_score = score

        if best_score >= threshold:
            matches.append(
                KeywordMatch(
                    job_keyword=job_keyword.phrase,
                    resume_keyword=best_match,
                    similarity=best_score,
                )
            )
        else:
            missing.append(job_keyword.phrase)

    coverage = len(matches) / len(job_keywords) if job_keywords else 0.0
    return matches, missing, coverage

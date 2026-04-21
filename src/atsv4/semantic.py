from __future__ import annotations

from .config import ATSConfig


def score_semantic_similarity(
    resume_sections: dict[str, str],
    job_text: str,
    config: ATSConfig,
) -> tuple[dict[str, float], float]:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers and scikit-learn are required for semantic similarity"
        ) from exc

    model = SentenceTransformer(config.embedding_model_name)
    section_scores: dict[str, float] = {}

    job_embedding = model.encode([job_text])

    for section_name, weight in config.semantic_section_weights.items():
        section_text = resume_sections.get(section_name, "").strip()
        if not section_text or weight <= 0:
            section_scores[section_name] = 0.0
            continue

        section_embedding = model.encode([section_text])
        score = float(cosine_similarity(section_embedding, job_embedding)[0][0])
        section_scores[section_name] = max(score, 0.0)

    weighted_score = sum(
        section_scores[name] * config.semantic_section_weights[name]
        for name in config.semantic_section_weights
    )
    return section_scores, min(weighted_score, 1.0)

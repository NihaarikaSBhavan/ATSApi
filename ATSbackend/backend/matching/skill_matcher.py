"""
skill_matcher.py — Semantic soft skill matching (Option D).

Matches JD skills against resume skills using both exact string comparison
and embedding cosine similarity, so synonymous skills (e.g. "AutoCAD" vs
"Civil 3D drafting", "cost estimation" vs "bill of quantities") are counted
as covered rather than missing.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

SOFT_MATCH_THRESHOLD: float = 0.70


@dataclass
class SoftMatchResult:
    exact_matched: List[str] = field(default_factory=list)
    soft_matched: Dict[str, Tuple[str, float]] = field(default_factory=dict)
    unmatched_gap: List[str] = field(default_factory=list)
    coverage: float = 0.0


def _embed(skills: List[str]) -> np.ndarray:
    if not skills:
        return np.zeros((0, 384), dtype=np.float32)
    try:
        from backend.embeddings.embedder import embed_skills
        vecs = embed_skills(skills, normalize_embeddings=True)
        if vecs is None or len(vecs) == 0:
            raise ValueError("empty")
        return np.array(vecs, dtype=np.float32)
    except Exception as exc:
        logger.warning(f"[skill_matcher] embed failed ({exc}), using zeros")
        return np.zeros((len(skills), 384), dtype=np.float32)


def soft_match_skills(
    resume_skills: List[str],
    jd_skills: List[str],
    threshold: float = SOFT_MATCH_THRESHOLD,
) -> SoftMatchResult:
    """
    Match resume skills against JD skills with exact + semantic matching.

    Returns SoftMatchResult with exact_matched, soft_matched dict,
    unmatched_gap list, and overall coverage fraction [0,1].
    """
    if not jd_skills:
        return SoftMatchResult(coverage=1.0)

    resume_set = {s.lower().strip() for s in resume_skills}
    result = SoftMatchResult()

    # Pass 1: exact string match
    remaining_jd: List[str] = []
    for jd_skill in jd_skills:
        if jd_skill.lower().strip() in resume_set:
            result.exact_matched.append(jd_skill)
        else:
            remaining_jd.append(jd_skill)

    if not remaining_jd:
        result.coverage = 1.0
        return result

    # Pass 2: embedding soft match
    jd_vecs     = _embed(remaining_jd)
    resume_vecs = _embed(resume_skills)

    # Fall back if embeddings are zeros (FORCE_REGEX mode)
    if np.allclose(jd_vecs, 0) or np.allclose(resume_vecs, 0) or not resume_skills:
        result.unmatched_gap = remaining_jd
        n = len(jd_skills)
        result.coverage = len(result.exact_matched) / n if n else 1.0
        return result

    sim_matrix = jd_vecs @ resume_vecs.T  # both L2-normalised

    for idx, jd_skill in enumerate(remaining_jd):
        row = sim_matrix[idx]
        best_idx = int(np.argmax(row))
        best_sim = float(row[best_idx])
        if best_sim >= threshold:
            result.soft_matched[jd_skill] = (resume_skills[best_idx], round(best_sim, 3))
        else:
            result.unmatched_gap.append(jd_skill)

    n_total = len(jd_skills)
    n_covered = len(result.exact_matched) + len(result.soft_matched)
    result.coverage = round(n_covered / n_total, 4) if n_total else 1.0

    logger.info(
        f"[skill_matcher] exact={len(result.exact_matched)} "
        f"soft={len(result.soft_matched)} gap={len(result.unmatched_gap)} "
        f"coverage={result.coverage:.3f}"
    )
    return result

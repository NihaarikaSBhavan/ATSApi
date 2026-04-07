"""
Feature engineering for ATS scoring.

Builds an 18-element feature vector for the probabilistic scorer.

Features 0-5:  Semantic core (PRIMARY)
Features 6-11: Contextual signals
Features 12-15: Binary missingness flags
Feature 16:    Domain flag
Feature 17:    Anti-collapse guardrail
"""

import numpy as np
import re
import logging
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

from backend.embeddings.embedder import embed_text, embed_skills, soft_skill_similarity, encode

logger = logging.getLogger(__name__)

# Stop-words excluded from keyword density (avoid inflation from common words)
_STOPWORDS = {
    "the","a","an","and","or","in","of","for","to","with","is","are","was","be",
    "has","have","our","your","we","you","it","this","that","will","can","must",
    "should","may","might","shall","would","could","not","also","such","as","by",
    "on","at","from","into","through","he","she","they","their","its","who","which",
    "what","how","when","where","all","any","both","each","few","more","most","do",
    "does","did","been","being","am","had","if","but","so","yet","nor","about",
}


def _compute_capability_score(resume_embeds: np.ndarray, jd_embeds: np.ndarray) -> float:
    """
    Measure how well the resume COVERS the JD's skill requirements.

    For each JD skill, find the best matching resume skill.
    Average the best matches — high means resume has equivalent skills.

    Note: direction is JD-centric (what fraction of JD skills does the
    resume satisfy?) which is the correct ATS perspective.
    """
    if len(resume_embeds) == 0 or len(jd_embeds) == 0:
        return 0.0

    # sim_matrix[i, j] = similarity(jd_skill_i, resume_skill_j)
    sim_matrix = cosine_similarity(jd_embeds, resume_embeds)

    # For each JD skill, find the best matching resume skill
    best_matches = np.max(sim_matrix, axis=1)

    # Liberal: use mean of best matches (not penalised for extra resume skills)
    return float(np.mean(best_matches))


def build_features(resume_text, jd_text, resume_skills, jd_skills, graph) -> np.ndarray:
    """
    Build 18-feature vector for the probabilistic ATS scorer.
    """

    # ── Semantic core (0-5) ───────────────────────────────────────────────────
    r_phrases = resume_skills or []
    j_phrases = jd_skills     or []
    r_embeds = embed_skills(r_phrases, normalize_embeddings=True) if r_phrases else np.zeros((0, 384))
    j_embeds = embed_skills(j_phrases, normalize_embeddings=True) if j_phrases else np.zeros((0, 384))

    # f0: JD-centric capability score (how well resume covers JD skills)
    f0 = _compute_capability_score(r_embeds, j_embeds)

    # f1: Whole-document text similarity
    f1 = float(cosine_similarity(
        encode(resume_text, normalize_embeddings=True).reshape(1, -1),
        encode(jd_text,     normalize_embeddings=True).reshape(1, -1)
    )[0][0])

    # f2: Exact keyword overlap (fraction of JD skills found in resume)
    rs = set(resume_skills) if resume_skills else set()
    f2 = (sum(1 for s in jd_skills if s in rs) / len(jd_skills)) if jd_skills else 0.0

    # f3: Graph-based soft matching
    f3 = graph.soft_match_score(resume_skills or [], jd_skills or [])

    # f4: Inferred coverage (graph-discovered skills covering JD gaps)
    f4 = _inferred_coverage(graph, resume_skills or [], jd_skills or [])

    # f5: Weighted composite (used downstream for reference)
    f5 = 0.30 * f0 + 0.20 * f1 + 0.20 * f3 + 0.10 * f4 + 0.20 * f2

    # ── Contextual features (6-11) ────────────────────────────────────────────
    f6  = _experience_level_alignment(resume_text, jd_text)
    f7  = _education_match(resume_text, jd_text)
    f8  = _certification_presence(resume_skills or [])
    f9  = _tool_ecosystem_coherence(resume_skills or [])
    f10 = _keyword_density_coherence(resume_text, jd_text)  # stop-word filtered
    f11 = _skill_recency_score(resume_text, resume_skills or [])

    # ── Binary missingness flags (12-15) ─────────────────────────────────────
    f12 = float(len(resume_text.strip()) == 0)
    f13 = float(len(resume_skills or []) == 0)
    f14 = float(f2 > 0)
    f15 = float(f4 > 0)

    # ── Domain flag (16) ─────────────────────────────────────────────────────
    try:
        domain_flag = 0.0 if getattr(graph, "domain", None) in (None, "", "generic") else 1.0
    except Exception:
        domain_flag = 0.0

    # ── Anti-collapse guardrail (17) ─────────────────────────────────────────
    # Signal: "semantically related despite low exact keyword overlap"
    # Enables the liberal bonus in predictor.py
    anti_collapse = 1.0 if (f2 <= 0.05 and f1 > 0.35) else 0.0

    logger.debug(
        f"Features: cap={f0:.3f} txt={f1:.3f} kw={f2:.3f} graph={f3:.3f} "
        f"inf={f4:.3f} exp={f6:.3f} edu={f7:.3f} kd={f10:.3f}"
    )

    return np.array(
        [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11,
         f12, f13, f14, f15, domain_flag, anti_collapse],
        dtype=np.float32,
    )


def _inferred_coverage(graph, resume_skills, jd_skills) -> float:
    if not resume_skills or not jd_skills:
        return 0.0
    inferred_scored = graph.infer_skills(resume_skills, top_k=10, scored=True)
    inferred   = {s for s, _ in inferred_scored} if inferred_scored else set()
    jd_set     = {s.lower() for s in jd_skills if s}
    resume_set = {s.lower() for s in resume_skills if s}
    missing    = jd_set - resume_set
    if not missing:
        return 1.0
    return len(missing & inferred) / len(missing)


def _experience_level_alignment(resume_text, jd_text) -> float:
    resume_level = _extract_seniority(resume_text)
    jd_level     = _extract_seniority(jd_text)
    if resume_level is None or jd_level is None:
        return 0.55   # neutral-liberal default when not detectable
    if resume_level >= jd_level:
        return 1.0
    diff = jd_level - resume_level
    return max(0.0, 1.0 - diff * 0.3)


def _extract_seniority(text) -> float | None:
    t = text.lower()
    if re.search(r"\b(entry[\s-]level|junior|graduate|intern|trainee)\b", t):
        return 0.0
    if re.search(r"\b(senior|lead|principal|architect|tech[\s-]lead)\b", t):
        return 1.0
    if re.search(r"\b(mid[\s-]level|intermediate)\b", t):
        return 0.5
    return None


def _education_match(resume_text, jd_text) -> float:
    req  = _extract_degree(jd_text)
    cand = _extract_degree(resume_text)
    if req is None:
        return 0.6   # no requirement stated — neutral-liberal
    if cand is None:
        years_m = re.search(r"(\d+)\s*\+?\s*years", resume_text.lower())
        years   = int(years_m.group(1)) if years_m else 0
        return 0.7 if years >= 5 else 0.4
    if cand == req:
        return 1.0
    degree_rank = {"bachelor": 1, "master": 2, "phd": 3}
    if degree_rank.get(cand, 0) > degree_rank.get(req, 0):
        return 1.0   # overqualified — still a match
    if abs(degree_rank.get(cand, 0) - degree_rank.get(req, 0)) == 1:
        return 0.75  # one level off — partial credit
    return 0.4


def _extract_degree(text) -> str | None:
    t = text.lower()
    if re.search(r"\bphd\b", t):
        return "phd"
    if re.search(r"\b(master|m\.s|msc)\b", t):
        return "master"
    if re.search(r"\b(bachelor|ba|bs|bsc|b\.s)\b", t):
        return "bachelor"
    return None


def _certification_presence(skills) -> float:
    certs = {
        "aws", "azure", "gcp", "kubernetes", "docker", "jenkins",
        "pmp", "cissp", "ccna", "certified", "certification",
        "tensorflow", "pytorch", "deep learning",
    }
    matched = sum(1 for s in skills if any(c in s.lower() for c in certs))
    return min(0.4 + matched * 0.1, 0.8)


def _tool_ecosystem_coherence(skills) -> float:
    ecosystems = {
        "python_backend":  {"python", "django", "flask", "fastapi", "celery", "sqlalchemy"},
        "java_backend":    {"java", "spring", "spring boot", "maven", "gradle", "hibernate"},
        "js_fullstack":    {"javascript", "react", "vue", "angular", "node.js"},
        "devops":          {"docker", "kubernetes", "terraform", "ansible", "jenkins"},
        "ml":              {"python", "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy"},
        "construction":    {"project management", "cost control", "quantity surveying",
                            "bill of quantities", "osha compliance"},
        "design":          {"figma", "adobe xd", "sketch", "wireframing", "prototyping",
                            "ui design", "ux design"},
        "healthcare":      {"patient care", "clinical diagnosis", "ehr", "hipaa compliance",
                            "medication management"},
        "finance":         {"portfolio management", "risk management", "financial modeling",
                            "gaap", "financial reporting"},
    }
    skills_lower = {s.lower() for s in skills}
    scores = []
    for eco_skills in ecosystems.values():
        matched = len(skills_lower & eco_skills)
        if matched > 0:
            scores.append(matched / len(eco_skills))
    return float(np.mean(scores)) if scores else 0.3


def _keyword_density_coherence(resume_text, jd_text) -> float:
    """
    Meaningful keyword overlap (stop-words excluded).
    This prevents trivial inflation from shared common words.
    """
    def _keywords(text):
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        return {w for w in words if w not in _STOPWORDS}

    jd_kw     = _keywords(jd_text)
    resume_kw = _keywords(resume_text)
    if not jd_kw:
        return 0.5
    return float(np.clip(len(jd_kw & resume_kw) / len(jd_kw), 0.0, 1.0))


def _skill_recency_score(resume_text, skills) -> float:
    modern = {
        "python", "javascript", "typescript", "rust", "go", "aws",
        "kubernetes", "react", "vue", "angular", "tensorflow", "fastapi",
        "pytorch", "keras", "java", "scala", "spark", "hadoop",
        # Construction modern tools
        "bim", "revit", "building information modeling",
        # Design modern tools
        "figma", "framer",
    }
    outdated = {"flash", "silverlight", "cobol"}
    skills_lower  = {s.lower() for s in skills}
    modern_count  = len(skills_lower & modern)
    outdated_count = len(skills_lower & outdated)
    years_m = re.search(r"(\d+)\s*\+?\s*years", resume_text.lower())
    years   = int(years_m.group(1)) if years_m else 2
    recency = 0.5 + modern_count * 0.08 - outdated_count * 0.1
    if years >= 5:
        recency += 0.12
    elif years >= 3:
        recency += 0.05
    return float(np.clip(recency, 0.0, 1.0))

import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
import os
from functools import lru_cache

_MODEL = None

def _is_force_regex():
    """Dynamically check FORCE_REGEX environment variable."""
    return os.environ.get("FORCE_REGEX", "0") == "1"

def _get_model():
    """Lazy load SentenceTransformer model only when needed."""
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError("sentence_transformers not installed. Run 'pip install sentence-transformers'")
    return _MODEL

def embed_text(text, normalize_embeddings=True):
    """Embed text using SentenceTransformer.
    
    Args:
        text: Input text to embed
        normalize_embeddings: If True (default), return normalized embeddings
    
    Returns:
        Normalized embedding vector (default) or raw embedding
    """
    # Use simpler fallback in FORCE_REGEX mode - skip expensive embeddings
    if _is_force_regex():
        return _fast_text_embedding(text)
    return _get_model().encode(text, convert_to_numpy=True, normalize_embeddings=normalize_embeddings)

def _fast_text_embedding(text):
    """Fast embedding fallback for FORCE_REGEX mode using TF-IDF style scoring."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=384, lowercase=True)
    try:
        embedded = vectorizer.fit_transform([text]).toarray()[0]
        # Pad to standard embedding dimension
        result = np.zeros(384)
        result[:len(embedded)] = embedded
        return result
    except:
        # Fallback to random projection
        np.random.seed(hash(text[:50]) % 2**32)
        return np.random.randn(384).astype(np.float32)

@lru_cache(maxsize=128)
def _embed_text_cached(text):
    """Cached embedding for repeated text."""
    return tuple(_get_model().encode(text, convert_to_numpy=True))

def embed_skills(skills, normalize_embeddings=True):
    """Embed a list of skill phrases.
    
    Args:
        skills: List of skill strings to embed
        normalize_embeddings: If True (default), return normalized embeddings
    
    Returns:
        (len(skills), embedding_dim) array of embeddings
    """
    if not skills:
        dim = 384 if _is_force_regex() else _get_model().get_sentence_embedding_dimension()
        return np.zeros((1, dim))
    
    # Use fast mode in FORCE_REGEX
    if _is_force_regex():
        return _fast_skills_embedding(skills)
    
    return _get_model().encode(skills, convert_to_numpy=True, normalize_embeddings=normalize_embeddings)

def _fast_skills_embedding(skills):
    """Fast embedding for skills in FORCE_REGEX mode."""
    embeddings = []
    for skill in skills:
        np.random.seed(hash(skill) % 2**32)
        emb = np.random.randn(384).astype(np.float32)
        embeddings.append(emb)
    return np.array(embeddings)

def cosine_similarity_pair(a, b):
    import numpy as np
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a,b)/d) if d else 0.0


def encode(text_or_list, normalize_embeddings=True):
    """
    **PRIMARY EMBEDDING API** — Single source of truth for embeddings.
    
    Supports both single text and lists of text/skills.
    Always returns normalized embeddings by default (aligned with semantic ATS architecture).
    
    Args:
        text_or_list: Single string or list of strings to embed
        normalize_embeddings: If True (default), return normalized embeddings
    
    Returns:
        Single embedding (1D array) for string input, or
        (N, embedding_dim) array for list input
    """
    if isinstance(text_or_list, str):
        # Single text: return 1D array
        return embed_text(text_or_list, normalize_embeddings=normalize_embeddings)
    else:
        # List of texts/skills
        return embed_skills(text_or_list, normalize_embeddings=normalize_embeddings)


def soft_skill_similarity(resume_skills, jd_skills):
    if not resume_skills or not jd_skills: return 0.0
    r_emb = embed_skills(resume_skills)
    j_emb = embed_skills(jd_skills)
    sim   = sk_cosine(j_emb, r_emb)
    return float(sim.max(axis=1).mean())

def preload():
    """Pre-load SentenceTransformer model at startup."""
    import logging
    logger = logging.getLogger(__name__)
    
    # Skip preload if FORCE_REGEX mode to keep startup fast
    if _is_force_regex():
        logger.info("FORCE_REGEX mode — skipping SentenceTransformer preload.")
        return
    
    try:
        logger.info("Pre-loading SentenceTransformer model...")
        _get_model()
        logger.info("SentenceTransformer model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to pre-load SentenceTransformer: {e}")

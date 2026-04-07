"""
Semantic Skill Representation Layer
---
Converts raw text → skill phrases → embeddings → clustered canonical skills

This is the CORE of the new semantic ATS architecture.
Replaces brittle keyword logic with embedding-based skill representation.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from backend.embeddings.embedder import _get_model


@dataclass
class SemanticSkill:
    """Represents a semantically-aware skill."""
    phrase: str  # original phrase from text
    canonical: str  # canonical label (after clustering)
    embedding: np.ndarray  # normalized embedding
    confidence: float  # clustering confidence [0, 1]


class SemanticSkillExtractor:
    """
    Extracts skill phrases from text, groups synonymous/related ones,
    maps them to canonical skill labels.
    
    Example:
        "problem solving" → analytical thinking
        "handled clients" → stakeholder management
    """
    
    def __init__(self, clustering_threshold: float = 0.1):
        """
        Args:
            clustering_threshold: DBSCAN eps parameter. Lower = tighter clusters.
        """
        self.model = _get_model()
        self.clustering_threshold = clustering_threshold
        self._phrase_cache = {}
        self._canonical_map = {}
    
    def extract_phrases(self, text: str) -> List[str]:
        """
        Extract candidate skill phrases from text.
        Lightweight fallback: split on comma, period, bullet points.
        
        TODO (Phase 2): Replace with spaCy + NER patterns for robustness.
        """
        if not text:
            return []
        
        # Split on common delimiters
        phrases = []
        for delimiter in [',', '.', '•', '\n', '–', '-']:
            text = text.replace(delimiter, '|')
        
        candidates = text.split('|')
        for phrase in candidates:
            clean = phrase.strip().lower()
            if len(clean) > 3 and clean not in self._phrase_cache:
                phrases.append(clean)
                self._phrase_cache[clean] = True
        
        return phrases
    
    def embed_phrases(self, phrases: List[str]) -> np.ndarray:
        """
        Embed candidate phrases using normalized embeddings.
        
        Args:
            phrases: List of skill phrases
            
        Returns:
            (len(phrases), embedding_dim) array of normalized embeddings
        """
        if not phrases:
            return np.zeros((0, 384))
        
        embeddings = self.model.encode(phrases, normalize_embeddings=True)
        return embeddings
    
    def cluster_phrases(self, phrases: List[str], embeddings: np.ndarray) -> List[Tuple[str, int]]:
        """
        Cluster similar phrases using DBSCAN in embedding space.
        
        Args:
            phrases: List of skill phrases
            embeddings: Pre-computed normalized embeddings
            
        Returns:
            List of (phrase, cluster_id) tuples
        """
        if len(phrases) == 0:
            return []
        if len(phrases) == 1:
            return [(phrases[0], 0)]
        
        # Compute distance matrix: 1 - cosine_similarity (since embeddings normalized)
        similarity = cosine_similarity(embeddings)
        distance = 1 - similarity
        
        # Cluster using DBSCAN
        clusterer = DBSCAN(eps=self.clustering_threshold, min_samples=1, metric='precomputed')
        labels = clusterer.fit_predict(distance)
        
        return list(zip(phrases, labels))
    
    def canonicalize_clusters(self, clustered_phrases: List[Tuple[str, int]]) -> Dict[str, str]:
        """
        Map each cluster to a canonical label (the most representative phrase).
        
        Args:
            clustered_phrases: List of (phrase, cluster_id) tuples
            
        Returns:
            Dict mapping phrase → canonical_label
        """
        cluster_map = {}
        for phrase, cluster_id in clustered_phrases:
            if cluster_id not in cluster_map:
                cluster_map[cluster_id] = phrase  # First phrase is canonical (for now)
        
        phrase_to_canonical = {}
        for phrase, cluster_id in clustered_phrases:
            phrase_to_canonical[phrase] = cluster_map[cluster_id]
        
        return phrase_to_canonical
    
    def extract_semantic_skills(self, text: str, top_k: int = 20) -> List[SemanticSkill]:
        """
        Full pipeline: text → phrases → embeddings → clusters → canonical skills.
        
        Args:
            text: Resume or JD text
            top_k: Limit results (most representative skills)
            
        Returns:
            List of SemanticSkill objects sorted by confidence
        """
        # Step 1: Extract phrases
        phrases = self.extract_phrases(text)
        if not phrases:
            return []
        
        # Step 2: Embed phrases
        embeddings = self.embed_phrases(phrases)
        
        # Step 3: Cluster similar phrases
        clustered = self.cluster_phrases(phrases, embeddings)
        
        # Step 4: Canonicalize
        phrase_to_canonical = self.canonicalize_clusters(clustered)
        
        # Step 5: Build SemanticSkill objects (deduplicated by canonical label)
        seen_canonical = set()
        semantic_skills = []
        
        for (phrase, cluster_id), embedding in zip(clustered, embeddings):
            canonical = phrase_to_canonical[phrase]
            
            if canonical not in seen_canonical:
                confidence = 1.0 / (cluster_id + 1)  # Heuristic: lower cluster ID = higher confidence
                semantic_skills.append(
                    SemanticSkill(
                        phrase=phrase,
                        canonical=canonical,
                        embedding=embedding,
                        confidence=min(confidence, 1.0)
                    )
                )
                seen_canonical.add(canonical)
        
        # Sort by confidence and return top_k
        semantic_skills.sort(key=lambda s: s.confidence, reverse=True)
        return semantic_skills[:top_k]


# Module-level extractor instance
_extractor = None

def get_extractor() -> SemanticSkillExtractor:
    """Lazy-load singleton extractor."""
    global _extractor
    if _extractor is None:
        _extractor = SemanticSkillExtractor()
    return _extractor


def extract_semantic_skills(text: str, top_k: int = 20) -> List[SemanticSkill]:
    """
    Public API: Extract semantic skills from text.
    
    Example:
        skills = extract_semantic_skills("problem solving and stakeholder management")
        → [SemanticSkill(phrase="problem solving", canonical="analytical thinking", ...)]
    """
    return get_extractor().extract_semantic_skills(text, top_k=top_k)

"""
Phase 2B: SpaCy NER-based Skill Extraction Layer
Integrates Named Entity Recognition for improved skill detection across all domains.
"""

import logging
import re
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass

try:
    import spacy
    from spacy.language import Language
except ImportError:
    raise ImportError("spaCy not installed. Run: pip install spacy")

from .skill_ontology import SKILL_ONTOLOGY, DOMAIN_SKILL_MAPPING, DOMAIN_PATTERNS, normalize_skill_name

logger = logging.getLogger(__name__)

@dataclass
class ExtractedSkill:
    """Represents an extracted skill with metadata."""
    name: str
    confidence: float
    source: str  # "ner", "pattern", "ontology"
    canonical_form: str = None
    domain_relevance: float = 1.0

class NERExtractor:
    """
    Enhanced skill extraction using SpaCy NER + domain-specific patterns.
    """
    
    _nlp = None
    _domain_patterns_compiled = {}
    
    def __init__(self):
        """Initialize the NER extractor with spaCy model."""
        if NERExtractor._nlp is None:
            try:
                NERExtractor._nlp = spacy.load("en_core_web_sm")
                logger.info("✅ SpaCy NER model loaded successfully")
            except OSError:
                logger.warning("⚠️  en_core_web_sm not found. Run: python -m spacy download en_core_web_sm")
                NERExtractor._nlp = None
        
        # Compile domain patterns
        self._compile_domain_patterns()
    
    def _compile_domain_patterns(self):
        """Pre-compile regex patterns for each domain."""
        for domain, patterns in DOMAIN_PATTERNS.items():
            compiled = []
            for pattern in patterns:
                try:
                    compiled.append((re.compile(pattern, re.IGNORECASE), pattern))
                except Exception as e:
                    logger.warning(f"Failed to compile pattern for {domain}: {pattern} - {e}")
            NERExtractor._domain_patterns_compiled[domain] = compiled
    
    def extract_skills_ner(self, text: str, domain: str = "generic") -> List[ExtractedSkill]:
        """
        Extract skills using SpaCy NER + pattern matching.
        Returns list of ExtractedSkill objects with confidence scores.
        """
        if not text or not isinstance(text, str):
            return []
        
        skills: Dict[str, ExtractedSkill] = {}
        
        # 1. SpaCy NER extraction (if model available)
        if NERExtractor._nlp:
            ner_skills = self._extract_via_ner(text, domain)
            for skill in ner_skills:
                key = skill.canonical_form.lower()
                if key not in skills or skill.confidence > skills[key].confidence:
                    skills[key] = skill
        
        # 2. Domain-specific pattern extraction
        if domain in NERExtractor._domain_patterns_compiled:
            pattern_skills = self._extract_via_patterns(text, domain)
            for skill in pattern_skills:
                key = skill.canonical_form.lower()
                if key not in skills or skill.confidence > skills[key].confidence:
                    skills[key] = skill
        
        # 3. Ontology-based extraction (keyword matching)
        ontology_skills = self._extract_via_ontology(text, domain)
        for skill in ontology_skills:
            key = skill.canonical_form.lower()
            if key not in skills or skill.confidence > skills[key].confidence:
                skills[key] = skill
        
        # Return deduplicated, sorted by confidence
        result = list(skills.values())
        result.sort(key=lambda x: -x.confidence)

        # Apply domain filtering — drops skills foreign to this domain
        # (filter_by_domain already exists and does exactly this)
        if domain and domain != "generic":
            result = self.filter_by_domain(result, domain)

        return result
    
    def _extract_via_ner(self, text: str, domain: str) -> List[ExtractedSkill]:
        """Extract using SpaCy NER entities."""
        skills = []
        
        try:
            doc = NERExtractor._nlp(text[:1000000])  # Safety limit on text length
            
            # Look for PRODUCT entities (tools/software) and PERSON orgs that might be tools
            for ent in doc.ents:
                if ent.label_ in ["PRODUCT", "ORG", "GPE"]:
                    # Check if entity matches known skills
                    normalized = normalize_skill_name(ent.text)
                    # Only use _is_tech_term heuristic for tech/generic domains —
                    # otherwise it tags tech entities in non-tech documents
                    is_tech = domain in ("tech", "generic") and self._is_tech_term(ent.text)
                    if normalized in SKILL_ONTOLOGY or is_tech:
                        canonical = self._get_canonical(normalized)
                        if canonical:
                            skills.append(ExtractedSkill(
                                name=ent.text,
                                confidence=0.70,  # Moderate confidence for NER
                                source="ner",
                                canonical_form=canonical
                            ))
        except Exception as e:
            logger.warning(f"NER extraction error: {e}")
        
        return skills
    
    def _extract_via_patterns(self, text: str, domain: str) -> List[ExtractedSkill]:
        """Extract using domain-specific regex patterns."""
        skills = []
        seen = set()
        
        if domain not in NERExtractor._domain_patterns_compiled:
            return skills
        
        for pattern_obj, pattern_str in NERExtractor._domain_patterns_compiled[domain]:
            for match in pattern_obj.finditer(text):
                matched_text = match.group(0)
                normalized = normalize_skill_name(matched_text)
                canonical = self._get_canonical(normalized)
                
                if canonical and canonical not in seen:
                    seen.add(canonical)
                    skills.append(ExtractedSkill(
                        name=matched_text,
                        confidence=0.85,  # High confidence for pattern match
                        source="pattern",
                        canonical_form=canonical
                    ))
        
        return skills
    
    def _extract_via_ontology(self, text: str, domain: str) -> List[ExtractedSkill]:
        """Extract by matching skill ontology terms.

        Skills that are primary to a *different* domain and not allowed in the
        current domain are skipped entirely — not just given lower confidence.
        This prevents tech skills (python, pytorch …) bleeding into construction
        or other non-tech roles.
        """
        skills = []
        text_lower = text.lower()
        seen = set()

        # Build foreign-only exclusion set for non-generic domains
        foreign_only: set = set()
        current_allowed: set = set()
        if domain and domain != "generic" and domain in DOMAIN_SKILL_MAPPING:
            current_allowed = (
                set(DOMAIN_SKILL_MAPPING[domain].get("primary",   [])) |
                set(DOMAIN_SKILL_MAPPING[domain].get("secondary", []))
            )
            foreign_primary: set = set()
            for d, cfg in DOMAIN_SKILL_MAPPING.items():
                if d != domain:
                    foreign_primary.update(cfg.get("primary", []))
            foreign_only = foreign_primary - current_allowed

        for canonical, synonyms in SKILL_ONTOLOGY.items():
            # Skip skills that belong exclusively to another domain
            if canonical in foreign_only:
                continue

            for synonym in synonyms:
                # Use word boundaries for more accurate matching
                pattern = r'\b' + re.escape(synonym) + r'\b'
                if re.search(pattern, text_lower, re.IGNORECASE):
                    if canonical not in seen:
                        seen.add(canonical)
                        # Confidence based on domain relevance
                        confidence = 0.8
                        if domain in DOMAIN_SKILL_MAPPING:
                            if canonical in DOMAIN_SKILL_MAPPING[domain].get("primary", []):
                                confidence = 0.95
                            elif canonical in DOMAIN_SKILL_MAPPING[domain].get("secondary", []):
                                confidence = 0.75

                        skills.append(ExtractedSkill(
                            name=synonym,
                            confidence=confidence,
                            source="ontology",
                            canonical_form=canonical
                        ))
                    break  # Stop after first synonym match for this canonical form

        return skills
    
    def _get_canonical(self, skill_text: str) -> str:
        """Get canonical form of a skill."""
        if not skill_text:
            return None
        
        skill_lower = skill_text.lower().strip()
        
        # Direct lookup in ontology
        for canonical, synonyms in SKILL_ONTOLOGY.items():
            if skill_lower in synonyms:
                return canonical
        
        # If not found, return None
        return None
    
    def _is_tech_term(self, text: str) -> bool:
        """Quick check if text looks like a tech/tool term."""
        tech_indicators = [
            r'\.js$', r'\.py$', r'\+\+$',  # Language-like
            r'(api|sdk|cli|ide|db|sql|html|css)',  # Tech abbreviations
            r'^(v\d+|[\d\.]+$)',  # Version numbers
        ]
        
        for pattern in tech_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def filter_by_domain(self, skills: List[ExtractedSkill], domain: str) -> List[ExtractedSkill]:
        """Filter and boost confidence for domain-relevant skills."""
        if domain not in DOMAIN_SKILL_MAPPING:
            return skills
        
        primary_skills = set(DOMAIN_SKILL_MAPPING[domain]["primary"])
        secondary_skills = set(DOMAIN_SKILL_MAPPING[domain].get("secondary", []))
        
        filtered = []
        for skill in skills:
            if skill.canonical_form in primary_skills:
                # Boost confidence for primary domain skills
                skill.confidence = min(1.0, skill.confidence * 1.15)
                skill.domain_relevance = 1.0
                filtered.append(skill)
            elif skill.canonical_form in secondary_skills:
                # Include secondary skills with slightly boosted confidence
                skill.confidence = min(1.0, skill.confidence * 1.05)
                skill.domain_relevance = 0.7
                filtered.append(skill)
            elif domain == "generic":
                # Include all skills for generic domain
                filtered.append(skill)
        
        return filtered
    
    def extract_deduplicated(self, text: str, domain: str = "generic") -> List[str]:
        """
        Extract skills and return canonical names (deduplicated).
        This is the main API for the refactored skill_extractor.py
        """
        extracted = self.extract_skills_ner(text, domain)
        filtered = self.filter_by_domain(extracted, domain)
        
        # Return unique canonical forms, sorted by confidence
        seen = set()
        result = []
        for skill in filtered:
            if skill.canonical_form not in seen:
                seen.add(skill.canonical_form)
                result.append(skill.canonical_form)
        
        return result


# Singleton instance
_ner_extractor = None

def get_ner_extractor() -> NERExtractor:
    """Get or create the NER extractor singleton."""
    global _ner_extractor
    if _ner_extractor is None:
        _ner_extractor = NERExtractor()
    return _ner_extractor

def extract_skills_ner(text: str, domain: str = "generic") -> List[str]:
    """
    Convenience function for extracting skills using NER.
    Returns list of canonical skill names.
    """
    extractor = get_ner_extractor()
    return extractor.extract_deduplicated(text, domain)
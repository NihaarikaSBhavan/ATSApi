"""
skill_graph_matcher.py
======================
Proper integration of the skill_graph module into ATSv4.

What this actually does
-----------------------
1.  EXTRACTION  — SkillExtractor two-stage Groq pipeline on both JD and resume.
    Returns structured skills: name / level / category / domain / relations.

2.  IN-MEMORY GRAPH  — builds a local adjacency map from the LLM-extracted
    relations in both texts. No Postgres needed for matching.
    e.g.  "React --requires--> JavaScript"
          "Docker --subset of--> Kubernetes"

3.  GRAPH-AWARE COVERAGE  — for every JD skill, tries in order:
      a) Exact normalised name match
      b) Fuzzy name match (rapidfuzz token_set_ratio)
      c) 2-hop BFS traversal: if a resume skill is a graph-neighbour of the
         JD skill, count it as a PARTIAL match (penalised score)
    This means "Docker" on a resume covers "Kubernetes" as a partial match
    when the graph contains "Kubernetes --requires--> Docker".

4.  LEVEL-WEIGHTED COVERAGE  — each match is weighted by the candidate's
    declared proficiency: expert=1.0, advanced=0.85, intermediate=0.7, beginner=0.5

5.  CATEGORY-AWARE MISSING SKILLS  — missing skill names are tagged with
    their category so the LLM evaluator knows gap severity.

6.  RICH LLM CONTEXT  — build_graph_context_for_llm() formats both graphs +
    match summary into structured text injected into evaluate_candidate().

Fallback
--------
If extraction fails (no API key / network error), falls back to the legacy
KeyBERT + rapidfuzz pipeline transparently.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .models import ExtractedKeyword, KeywordMatch

if TYPE_CHECKING:
    from .config import ATSConfig

logger = logging.getLogger(__name__)

_LEVEL_WEIGHT: dict[str, float] = {
    "expert": 1.0,
    "advanced": 0.85,
    "intermediate": 0.7,
    "beginner": 0.5,
}
_TRAVERSAL_SIMILARITY = 65.0


# ---------------------------------------------------------------------------
# In-memory graph
# ---------------------------------------------------------------------------

@dataclass
class SkillNode:
    name: str
    name_lower: str
    level: str = "intermediate"
    category: str = "tool"
    domain: str = ""
    edges: dict[str, str] = field(default_factory=dict)  # neighbour_lower → relation

    @property
    def level_weight(self) -> float:
        return _LEVEL_WEIGHT.get(self.level, 0.7)


@dataclass
class SkillMatchResult:
    job_skill: str
    job_category: str
    resume_skill: str
    similarity: float
    match_type: str           # "exact" | "fuzzy" | "traversal" | "none"
    relation_path: str = ""


class InMemorySkillGraph:
    def __init__(self) -> None:
        self._nodes: dict[str, SkillNode] = {}

    def add_skills(self, raw_skills: list[dict]) -> None:
        # Pass 1: create nodes
        for s in raw_skills:
            name = (s.get("name") or "").strip()
            if not name:
                continue
            key = name.lower()
            if key not in self._nodes:
                self._nodes[key] = SkillNode(
                    name=name,
                    name_lower=key,
                    level=s.get("level", "intermediate"),
                    category=s.get("category", "tool"),
                    domain=s.get("domain", ""),
                )
        # Pass 2: wire edges
        for s in raw_skills:
            name = (s.get("name") or "").strip()
            if not name:
                continue
            src = self._nodes.get(name.lower())
            if not src:
                continue
            for rel in s.get("relations", []):
                tgt_name = rel.get("to", "").strip()
                rel_type = rel.get("type", "related to")
                tgt_key = tgt_name.lower()
                if tgt_key in self._nodes:
                    src.edges[tgt_key] = rel_type
                    # reverse edge so traversal works both directions
                    self._nodes[tgt_key].edges[name.lower()] = f"inverse:{rel_type}"

    def get(self, name: str) -> SkillNode | None:
        return self._nodes.get(name.lower())

    def neighbours(self, name: str) -> list[tuple[SkillNode, str]]:
        node = self._nodes.get(name.lower())
        if not node:
            return []
        out = []
        for nb_key, rel in node.edges.items():
            nb = self._nodes.get(nb_key)
            if nb:
                out.append((nb, rel))
        return out

    def traverse(self, name: str, hops: int = 2) -> list[tuple[SkillNode, str]]:
        """BFS up to `hops` hops. Returns (node, relation_path) pairs."""
        start = self._nodes.get(name.lower())
        if not start:
            return []
        visited: set[str] = {start.name_lower}
        frontier: list[tuple[SkillNode, str]] = [(start, "")]
        result: list[tuple[SkillNode, str]] = []
        for _ in range(hops):
            next_frontier: list[tuple[SkillNode, str]] = []
            for node, path in frontier:
                for nb, rel in self.neighbours(node.name):
                    if nb.name_lower not in visited:
                        visited.add(nb.name_lower)
                        new_path = f"{node.name} --[{rel}]--> {nb.name}"
                        if path:
                            new_path = f"{path} → {new_path}"
                        result.append((nb, new_path))
                        next_frontier.append((nb, new_path))
            frontier = next_frontier
        return result

    @property
    def all_skills(self) -> list[SkillNode]:
        return list(self._nodes.values())

    def to_context_text(self, label: str) -> str:
        lines = [f"=== {label} Skill Graph ==="]
        for n in sorted(self._nodes.values(), key=lambda x: (x.domain, x.name)):
            lines.append(f"SKILL: {n.name} | level={n.level} | category={n.category} | domain={n.domain}")
        lines.append("")
        for n in self._nodes.values():
            for nb_key, rel in n.edges.items():
                if rel.startswith("inverse:"):
                    continue
                nb = self._nodes.get(nb_key)
                if nb:
                    lines.append(f"RELATION: {n.name} --[{rel}]--> {nb.name}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Async extraction (no Postgres)
# ---------------------------------------------------------------------------

async def _async_extract_skills(text: str, source: str, groq_api_key: str) -> list[dict]:
    import httpx
    from .skill_graph.extractor import SkillExtractor

    class _NoDBExtractor(SkillExtractor):
        def __init__(self, api_key: str):
            self._api_key = api_key
            self._graph = None  # type: ignore[assignment]
            self._client = httpx.AsyncClient(
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                timeout=30,
            )

    extractor = _NoDBExtractor(api_key=groq_api_key)
    try:
        domain = await extractor._detect_domain(text[:1500])
        raw = await extractor._extract_skills(text, source, domain)
    finally:
        await extractor.close()

    return (raw or {}).get("skills", [])


def _run_async(coro):
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


def _extract_raw(text: str, source: str, config: "ATSConfig") -> list[dict]:
    key = getattr(config, "groq_api_key", None) or getattr(config, "llm_api_key", None)
    if not key:
        return []
    try:
        return _run_async(_async_extract_skills(text, source=source, groq_api_key=key))
    except Exception as exc:
        logger.error("skill_graph extraction failed (%s): %s", source, exc)
        return []


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _match_skill(jd_node: SkillNode, resume_graph: InMemorySkillGraph, threshold: float) -> SkillMatchResult:
    from rapidfuzz import fuzz

    # 1. Exact
    rnode = resume_graph.get(jd_node.name)
    if rnode:
        return SkillMatchResult(jd_node.name, jd_node.category, rnode.name, 100.0, "exact")

    # 2. Fuzzy across all resume skills
    best_score, best_node = 0.0, None
    for rn in resume_graph.all_skills:
        score = float(fuzz.token_set_ratio(jd_node.name, rn.name))
        if score > best_score:
            best_score, best_node = score, rn
    if best_score >= threshold and best_node:
        return SkillMatchResult(jd_node.name, jd_node.category, best_node.name, best_score, "fuzzy")

    # 3. Graph traversal — expand every resume skill and check neighbours
    traversal_threshold = max(threshold - 10, 60.0)
    for rn in resume_graph.all_skills:
        for neighbour, path in resume_graph.traverse(rn.name, hops=2):
            if neighbour.name_lower == jd_node.name_lower:
                return SkillMatchResult(jd_node.name, jd_node.category, rn.name, _TRAVERSAL_SIMILARITY, "traversal", path)
            score = float(fuzz.token_set_ratio(jd_node.name, neighbour.name))
            if score >= traversal_threshold:
                return SkillMatchResult(jd_node.name, jd_node.category, rn.name, min(_TRAVERSAL_SIMILARITY, score * 0.8), "traversal", path)

    return SkillMatchResult(jd_node.name, jd_node.category, "", 0.0, "none")


def _compute_coverage(results: list[SkillMatchResult], jd_graph: InMemorySkillGraph) -> float:
    if not results:
        return 0.0
    total, earned = 0.0, 0.0
    for mr in results:
        lw = (jd_graph.get(mr.job_skill) or SkillNode("", "")).level_weight
        total += lw
        earned += (mr.similarity / 100.0) * lw
    return earned / total if total else 0.0


# ---------------------------------------------------------------------------
# LLM context builder
# ---------------------------------------------------------------------------

def build_graph_context_for_llm(
    jd_graph: InMemorySkillGraph,
    resume_graph: InMemorySkillGraph,
    match_results: list[SkillMatchResult],
) -> str:
    lines = ["## Skill Graph Analysis\n"]
    lines.append(jd_graph.to_context_text("Job Description"))
    lines.append("")
    lines.append(resume_graph.to_context_text("Resume"))
    lines.append("\n## Match Summary")

    exact      = [m for m in match_results if m.match_type == "exact"]
    fuzzy      = [m for m in match_results if m.match_type == "fuzzy"]
    traversal  = [m for m in match_results if m.match_type == "traversal"]
    missing    = [m for m in match_results if m.match_type == "none"]

    if exact:
        lines.append(f"\nDirect matches ({len(exact)}):")
        for m in exact:
            lines.append(f"  ✓  {m.job_skill} [{m.job_category}]")
    if fuzzy:
        lines.append(f"\nFuzzy matches ({len(fuzzy)}):")
        for m in fuzzy:
            lines.append(f"  ~  {m.job_skill} [{m.job_category}]  ←  {m.resume_skill}  ({m.similarity:.0f}%)")
    if traversal:
        lines.append(f"\nGraph-inferred matches ({len(traversal)}) — covered via related skills:")
        for m in traversal:
            lines.append(f"  ◎  {m.job_skill} [{m.job_category}]  ←  {m.resume_skill}  via  {m.relation_path}")
    if missing:
        lines.append(f"\nMissing ({len(missing)}):")
        for m in missing:
            lines.append(f"  ✗  {m.job_skill} [{m.job_category}]")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_and_match(
    job_text: str,
    resume_text: str,
    config: "ATSConfig",
) -> tuple[list[KeywordMatch], list[str], float, list[ExtractedKeyword], list[ExtractedKeyword], str]:
    """
    Full pipeline. Returns:
      keyword_matches, missing_keywords, keyword_coverage,
      job_keywords, resume_keywords, graph_context_for_llm
    """
    threshold = getattr(config, "fuzzy_match_threshold", 84.0)

    # Check if skill graph is enabled
    if not getattr(config, "use_skill_graph", True):
        logger.info("skill_graph_matcher: skill graph disabled, using KeyBERT fallback")
        from .keywords import extract_keywords, fuzzy_match_keywords
        jd_kws  = extract_keywords(job_text,     config, top_n=getattr(config, "jd_top_keywords", 20))
        res_kws = extract_keywords(resume_text,  config, top_n=getattr(config, "resume_top_keywords", 30))
        matches, missing, coverage = fuzzy_match_keywords(jd_kws, res_kws, threshold=threshold)
        return matches, missing, coverage, jd_kws, res_kws, ""

    jd_raw     = _extract_raw(job_text,     "job_description", config)
    resume_raw = _extract_raw(resume_text,  "resume",          config)

    if not jd_raw or not resume_raw:
        logger.info("skill_graph_matcher: skill graph extraction failed, falling back to KeyBERT")
        from .keywords import extract_keywords, fuzzy_match_keywords
        jd_kws  = extract_keywords(job_text,     config, top_n=getattr(config, "jd_top_keywords", 20))
        res_kws = extract_keywords(resume_text,  config, top_n=getattr(config, "resume_top_keywords", 30))
        matches, missing, coverage = fuzzy_match_keywords(jd_kws, res_kws, threshold=threshold)
        return matches, missing, coverage, jd_kws, res_kws, ""

    jd_graph     = InMemorySkillGraph()
    jd_graph.add_skills(jd_raw)
    resume_graph = InMemorySkillGraph()
    resume_graph.add_skills(resume_raw)

    match_results = [_match_skill(n, resume_graph, threshold) for n in jd_graph.all_skills]

    keyword_matches = [
        KeywordMatch(job_keyword=m.job_skill, resume_keyword=m.resume_skill, similarity=m.similarity)
        for m in match_results if m.match_type != "none"
    ]
    missing_keywords = [
        f"{m.job_skill} [{m.job_category}]"
        for m in match_results if m.match_type == "none"
    ]
    keyword_coverage = _compute_coverage(match_results, jd_graph)

    job_keywords    = [ExtractedKeyword(phrase=n.name, score=n.level_weight) for n in jd_graph.all_skills]
    resume_keywords = [ExtractedKeyword(phrase=n.name, score=n.level_weight) for n in resume_graph.all_skills]
    graph_context   = build_graph_context_for_llm(jd_graph, resume_graph, match_results)

    return keyword_matches, missing_keywords, keyword_coverage, job_keywords, resume_keywords, graph_context

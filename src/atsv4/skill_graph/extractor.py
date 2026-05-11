"""
SkillExtractor — Groq/Llama-powered skill extraction pipeline.

Two-stage pipeline:
  Stage 1 (llama-3.1-8b-instant)  : fast domain detection (~50 tokens)
  Stage 2 (llama-3.3-70b-versatile): full skill + relationship extraction

Extracted skills are automatically merged into the SkillGraph (Postgres).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import httpx

from .graph import SkillGraph
from .models import ExtractionResult, RawSkill, DOMAIN_HINTS

logger = logging.getLogger(__name__)

# ── Groq config ───────────────────────────────────────────────────────────────

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

FAST_MODEL = "llama-3.1-8b-instant"       # domain detection
SMART_MODEL = "llama-3.3-70b-versatile"   # full extraction

# Re-export for convenience — single source of truth lives in models.py
DOMAINS = DOMAIN_HINTS

SOURCE_MAP = {
    "job_description": "job description",
    "resume": "resume or CV",
    "linkedin": "LinkedIn profile",
    "manual": "skills list",
}


class SkillExtractor:
    """
    Groq-powered skill extractor that feeds directly into SkillGraph.

    Usage:
        extractor = SkillExtractor(api_key="gsk_...", graph=skill_graph)
        result = await extractor.extract_and_merge(text, source="job_description")
        print(result.detected_domain)
        print(result.new_skills_added)
    """

    def __init__(self, api_key: str, graph: SkillGraph, timeout: int = 30):
        self._api_key = api_key
        self._graph = graph
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    async def close(self):
        await self._client.aclose()

    # ── Public API ────────────────────────────────────────────────────────────

    async def extract_and_merge(
        self,
        text: str,
        source: str = "job_description",
    ) -> ExtractionResult:
        """
        Full pipeline:
          1. Detect domain (fast, 8B model)
          2. Extract skills + relationships (70B model)
          3. Merge into Postgres graph (dedup, upsert)
          4. Return ExtractionResult with stats
        """
        text = text.strip()[:4000]  # guard against huge inputs

        # Stage 1: domain detection
        detected_domain = await self._detect_domain(text)
        logger.info(f"Detected domain: {detected_domain}")

        # Stage 2: full extraction
        raw = await self._extract_skills(text, source, detected_domain)
        if raw is None:
            raise ValueError("LLM returned unparseable response — check Groq logs")

        # Stage 3: merge into graph
        stats = await self._graph.merge_skills(raw["skills"], source=source)

        return ExtractionResult(
            detected_domain=raw.get("detected_domain", detected_domain),
            secondary_domains=raw.get("secondary_domains", []),
            summary=raw.get("summary", ""),
            raw_skills=[RawSkill(**s) for s in raw["skills"]],
            new_skills_added=stats["new_skills_added"],
            new_edges_added=stats["new_edges_added"],
            total_skills_in_graph=stats["total_skills_in_graph"],
        )

    async def detect_domain_only(self, text: str) -> str:
        """
        Lightweight domain detection — use this at the start of ATS scoring
        to select the right subgraph before the full scoring call.
        """
        return await self._detect_domain(text[:1500])

    # ── Stage 1: Domain Detection ─────────────────────────────────────────────

    async def _detect_domain(self, text: str) -> str:
        domain_list = "\n".join(f"- {d}" for d in DOMAINS)
        prompt = f"""Classify this text into exactly one domain. Reply with ONLY the domain name, nothing else.

Domains:
{domain_list}

Text:
{text[:1000]}"""

        response = await self._call_groq(
            model=FAST_MODEL,
            messages=[
                {"role": "system", "content": "You are a domain classifier. Reply with exactly one domain name from the provided list. No explanation."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=20,
            temperature=0,
        )

        detected = (response or "").strip().strip('"').strip("'")

        # Fuzzy match against known domains
        for d in DOMAINS:
            if d.lower() in detected.lower() or detected.lower() in d.lower():
                return d
        return "Other"

    # ── Stage 2: Full Extraction ──────────────────────────────────────────────

    async def _extract_skills(
        self,
        text: str,
        source: str,
        detected_domain: str,
    ) -> Optional[dict]:
        source_label = SOURCE_MAP.get(source, source)
        domain_list = ", ".join(DOMAINS)

        prompt = f"""Extract all skills, tools, technologies, and competencies from this {source_label}.

Return ONLY a valid JSON object with this exact structure. No markdown. No explanation. Just JSON.

{{
  "detected_domain": "{detected_domain}",
  "secondary_domains": [],
  "summary": "<2 sentence summary>",
  "skills": [
    {{
      "name": "<skill name>",
      "domain": "<one of: {domain_list}>",
      "level": "<beginner|intermediate|advanced|expert>",
      "category": "<language|framework|tool|methodology|soft skill|certification|concept|platform>",
      "relations": [
        {{"to": "<other skill name from this list>", "type": "<requires|related to|subset of|leads to|alternative to|enhances>"}}
      ]
    }}
  ]
}}

Rules:
- Extract every skill mentioned, implied, or required
- Relations must only reference other skills in YOUR output list
- Level: infer from context (e.g. "expert Python" → expert, "familiar with Docker" → beginner)
- Be specific: "React" not "frontend", "PostgreSQL" not "database"

Text:
{text}"""

        raw_text = await self._call_groq(
            model=SMART_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert ATS skill extraction system. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4000,
            temperature=0,
            json_mode=True,
        )

        return self._parse_json(raw_text)

    # ── Groq HTTP Client ──────────────────────────────────────────────────────

    async def _call_groq(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 1000,
        temperature: float = 0,
        json_mode: bool = False,
    ) -> Optional[str]:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = await self._client.post(GROQ_API_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            logger.error(f"Groq API error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Groq call failed: {e}")
            raise

    # ── JSON Parser ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: Optional[str]) -> Optional[dict]:
        if not text:
            return None
        # Strip markdown fences if present
        clean = re.sub(r"^```json\s*", "", text.strip(), flags=re.IGNORECASE)
        clean = re.sub(r"^```\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"```\s*$", "", clean).strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            # Last resort: find the outermost JSON object
            match = re.search(r"\{[\s\S]*\}", clean)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        logger.error(f"Failed to parse JSON from LLM response: {clean[:300]}")
        return None
"""
FastAPI router for the skill graph module.

Mount into your existing FastAPI app:

    from src.router import create_router
    app.include_router(create_router(), prefix="/skills")

The router reads graph and extractor from request.app.state, which lifespan
populates at startup.

Endpoints:
    POST   /skills/extract           — extract from JD or resume
    GET    /skills                   — list all skills (optional ?domain=)
    GET    /skills/stats             — graph stats
    GET    /skills/subgraph/{domain} — domain subgraph + LLM context text
    GET    /skills/{id}/traverse     — BFS traversal from a skill
    DELETE /skills/{id}              — delete skill
    GET    /skills/export            — full ATS-ready JSON export
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from .graph import SkillGraph
from .extractor import SkillExtractor
from .models import ExtractionResult, RawSkill

logger = logging.getLogger(__name__)


# ── Request / Response models ─────────────────────────────────────────────────

class ExtractRequest(BaseModel):
    text: str
    # Pydantic will reject any value not in this set with a clear 422 response,
    # so invalid source strings can never reach the DB.
    source: Literal["job_description", "resume", "linkedin", "manual"] = "job_description"


class ExtractResponse(BaseModel):
    detected_domain: str
    secondary_domains: list[str]
    summary: str
    raw_skills: list[RawSkill]
    skills_extracted: int
    new_skills_added: int
    new_edges_added: int
    total_skills_in_graph: int


# ── Router factory ────────────────────────────────────────────────────────────

def create_router() -> APIRouter:
    """
    Return a fully configured APIRouter.  graph and extractor are resolved
    from request.app.state at call time, so this function can be called at
    module import time — before lifespan has run.
    """
    router = APIRouter(tags=["Skill Graph"])

    @router.post("/extract", response_model=ExtractResponse)
    async def extract_skills(req: ExtractRequest, request: Request):
        """
        Extract skills from a job description or resume and merge into graph.
        Automatically detects domain, deduplicates, and creates relationships.
        """
        if not req.text.strip():
            raise HTTPException(400, "Text cannot be empty")

        extractor: SkillExtractor = request.app.state.extractor
        try:
            result: ExtractionResult = await extractor.extract_and_merge(
                req.text, source=req.source
            )
        except asyncio.CancelledError:
            # Request was cancelled by the client — let it propagate naturally.
            raise
        except ValueError as e:
            raise HTTPException(422, str(e))
        except Exception as e:
            # Log the full traceback so the real cause is never silently lost,
            # then surface a safe message to the caller.
            logger.exception("Extraction failed for source=%s", req.source)
            raise HTTPException(500, f"Extraction failed: {type(e).__name__}")

        return ExtractResponse(
            detected_domain=result.detected_domain,
            secondary_domains=result.secondary_domains,
            summary=result.summary,
            raw_skills=result.raw_skills,
            skills_extracted=len(result.raw_skills),
            new_skills_added=result.new_skills_added,
            new_edges_added=result.new_edges_added,
            total_skills_in_graph=result.total_skills_in_graph,
        )

    @router.get("/")
    async def list_skills(request: Request, domain: Optional[str] = Query(None)):
        """List all skills, optionally filtered by domain."""
        graph: SkillGraph = request.app.state.graph
        skills = await graph.get_all_skills(domain=domain)
        return {"skills": [s.model_dump() for s in skills], "count": len(skills)}

    @router.get("/stats")
    async def get_stats(request: Request):
        """Graph statistics by domain."""
        graph: SkillGraph = request.app.state.graph
        return await graph.stats()

    @router.get("/subgraph/{domain}")
    async def get_subgraph(domain: str, request: Request):
        """
        Get all skills + edges for a domain, plus a pre-built context_text
        string ready to inject into your Groq/Llama scoring prompt.
        """
        graph: SkillGraph = request.app.state.graph
        subgraph = await graph.get_domain_subgraph(domain)
        if not subgraph["skills"]:
            raise HTTPException(404, f"No skills found for domain: {domain}")
        return subgraph

    @router.get("/export")
    async def export_graph(request: Request):
        """Full ATS-ready JSON export of the entire skill graph."""
        graph: SkillGraph = request.app.state.graph
        return await graph.export_ats_json()

    @router.get("/{skill_id}/traverse")
    async def traverse_skill(
        skill_id: str,
        request: Request,
        hops: int = Query(2, ge=1, le=4),
    ):
        """BFS traversal from a skill node. Returns all skills within N hops."""
        graph: SkillGraph = request.app.state.graph
        skill = await graph.get_skill_by_id(skill_id)
        if not skill:
            raise HTTPException(404, "Skill not found")
        related = await graph.traverse(skill.name, hops=hops)
        return {
            "root": skill.model_dump(),
            "related": [s.model_dump() for s in related if s.id != skill.id],
            "hops": hops,
        }

    @router.delete("/{skill_id}")
    async def delete_skill(skill_id: str, request: Request):
        """Delete a skill and all its edges."""
        graph: SkillGraph = request.app.state.graph
        skill = await graph.get_skill_by_id(skill_id)
        if not skill:
            raise HTTPException(404, "Skill not found")
        await graph.delete_skill(skill_id)
        return {"deleted": skill_id, "name": skill.name}

    return router
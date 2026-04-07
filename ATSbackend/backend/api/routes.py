"""API routes for ATS engine."""
import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from backend.parser.document_parser import parse_document
from backend.extraction.skill_extractor import extract_skills, get_model_info, filter_relevant_skills
from backend.graph.skill_graph import SkillGraph, detect_domain
from backend.features.feature_engineer import build_features
from backend.scoring.predictor import predict_score, score_label
from backend.matching.skill_matcher import soft_match_skills

from .models import ATSResult
from .config import get_max_upload_size, get_frontend_dir
from .middleware import setup_rate_limiting

logger = logging.getLogger(__name__)
MAX_UPLOAD_SIZE_BYTES = get_max_upload_size()
FRONTEND_DIR = get_frontend_dir()

router = APIRouter()


# ─── Helpers ────────────────────────────────────────────────────────────────


async def _validate_file(file_obj: UploadFile, file_type: str) -> bytes:
    data = await file_obj.read()
    if len(data) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            413,
            f"{file_type} exceeds {MAX_UPLOAD_SIZE_BYTES / (1024*1024):.0f}MB limit"
        )
    if len(data) == 0:
        raise HTTPException(400, f"{file_type} file is empty")
    return data


async def _parse_jd_upload(jd_file_obj) -> str:
    data = await jd_file_obj.read()
    tmp = None
    try:
        suffix = Path(jd_file_obj.filename or "jd.txt").suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(data)
        tmp.close()
        return await asyncio.to_thread(parse_document, tmp.name)
    except Exception as e:
        logger.error(f"JD parse error: {e}", exc_info=True)
        raise HTTPException(400, "Failed to parse job description file.")
    finally:
        if tmp and os.path.exists(tmp.name):
            os.unlink(tmp.name)


async def _run_pipeline(
    resume_bytes: bytes,
    resume_filename: str,
    jd_text: str,
    progress,
    domain: str = None,
) -> ATSResult:
    """Full ATS analysis pipeline."""

    # ── 1. Parse resume ──────────────────────────────────────────────────────
    await progress("parse_resume", "Parsing resume document...", 10)
    tmp = None
    try:
        suffix = Path(resume_filename or "resume.pdf").suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(resume_bytes)
        tmp.close()
        resume_text = await asyncio.to_thread(parse_document, tmp.name)
    except Exception as e:
        logger.error(f"Resume parse error: {e}", exc_info=True)
        raise HTTPException(400, "Failed to parse resume file.")
    finally:
        if tmp and os.path.exists(tmp.name):
            os.unlink(tmp.name)

    # ── 2. Detect domain from JD BEFORE extraction ───────────────────────────
    # Domain must be known first so extract_skills() suppresses cross-domain
    # phantom skills (e.g. python/pytorch in a construction JD).
    await progress("detect_domain", "Detecting job domain...", 20)
    detected_domain = domain if domain else await asyncio.to_thread(detect_domain, jd_text)
    logger.info(f"[pipeline] domain={detected_domain}")

    # ── 3. Extract skills (domain-aware) ─────────────────────────────────────
    await progress("extract_resume", "Extracting skills from resume...", 33)
    resume_skills = await asyncio.to_thread(extract_skills, resume_text, detected_domain)

    await progress("extract_jd", "Extracting skills from job description...", 48)
    jd_skills = await asyncio.to_thread(extract_skills, jd_text, detected_domain)

    # Second-pass domain filter: removes any cross-domain stragglers
    resume_skills = filter_relevant_skills(resume_skills, domain=detected_domain)
    jd_skills     = filter_relevant_skills(jd_skills,     domain=detected_domain)

    logger.info(f"[pipeline] resume_skills={len(resume_skills)} jd_skills={len(jd_skills)}")

    # ── 4. Build skill graph ──────────────────────────────────────────────────
    await progress("skill_graph", "Building skill graph...", 58)
    graph = SkillGraph(domain=detected_domain)
    await asyncio.to_thread(graph.build, [resume_skills, jd_skills])

    # ── 5. Build feature vector ───────────────────────────────────────────────
    await progress("features", "Computing semantic features...", 70)
    features = await asyncio.to_thread(
        build_features, resume_text, jd_text, resume_skills, jd_skills, graph
    )

    # ── 6. Score ──────────────────────────────────────────────────────────────
    await progress("predict", "Scoring (probabilistic model)...", 82)
    score = predict_score(features)
    label = score_label(score)

    # ── 7. Soft/semantic skill matching ──────────────────────────────────────
    await progress("soft_match", "Running semantic skill matching...", 91)
    match_result = await asyncio.to_thread(soft_match_skills, resume_skills, jd_skills)

    await progress("done", f"Complete — {label} ({score})", 100)

    # ── Unpack features for response ─────────────────────────────────────────
    def f(i, d=0.0):
        return round(float(features[i]), 3) if len(features) > i else d

    semantic_features = {
        "capability_score":     f(0),
        "embedding_similarity": f(1),
        "keyword_overlap":      f(2),
        "graph_match_score":    f(3),
        "inferred_coverage":    f(4),
        "composite_score":      f(5),
    }
    contextual_features = {
        "experience_alignment": f(6),
        "education_match":      f(7),
        "certifications":       f(8),
        "ecosystem_coherence":  f(9),
        "keyword_density":      f(10),
        "skill_recency":        f(11),
    }

    # Inferred skills from graph
    inferred_scored = graph.infer_skills(resume_skills, top_k=8, scored=True)
    inferred      = [s for s, _ in inferred_scored]
    inferred_conf = {s: float(round(sc, 3)) for s, sc in inferred_scored}
    gap           = graph.skill_gap_analysis(resume_skills, jd_skills)

    explanation = _explain(score, semantic_features, match_result)

    soft_serialised = {
        jd_s: [res_s, sim]
        for jd_s, (res_s, sim) in match_result.soft_matched.items()
    }

    return ATSResult(
        score=score,
        label=label,
        resume_skills=sorted(resume_skills),
        jd_skills=sorted(jd_skills),
        matched_skills=sorted(match_result.exact_matched),
        soft_matched_skills=soft_serialised,
        missing_skills=sorted(match_result.unmatched_gap),
        skill_coverage=match_result.coverage,
        inferred_skills=inferred,
        inferred_skill_confidences=inferred_conf,
        skill_gap=gap,
        domain=detected_domain,
        features={
            **{f"semantic.{k}": v for k, v in semantic_features.items()},
            **{f"contextual.{k}": v for k, v in contextual_features.items()},
            # Legacy keys for backward compatibility
            "global_similarity":    semantic_features["embedding_similarity"],
            "skill_embedding_soft": semantic_features["capability_score"],
            "keyword_overlap":      semantic_features["keyword_overlap"],
            "graph_soft_match":     semantic_features["graph_match_score"],
            "inferred_coverage":    semantic_features["inferred_coverage"],
            "weighted_composite":   semantic_features["composite_score"],
            "soft_skill_coverage":  match_result.coverage,
        },
        explanation=explanation,
    )


def _explain(score: float, sf: dict, match_result) -> str:
    parts = []
    cap = sf.get("capability_score", 0)
    emb = sf.get("embedding_similarity", 0)

    if cap > 0.7:
        parts.append("✓ Strong semantic skill alignment")
    elif cap > 0.45:
        parts.append("~ Moderate semantic skill alignment")
    else:
        parts.append("⚠ Limited semantic skill overlap")

    if emb > 0.55:
        parts.append("✓ Resume and JD are semantically related")
    elif emb > 0.35:
        parts.append("~ Resume and JD share some conceptual overlap")

    exact  = len(match_result.exact_matched)
    soft   = len(match_result.soft_matched)
    cov    = round(match_result.coverage * 100)
    if soft > 0:
        parts.append(f"✓ {exact} exact + {soft} semantic skill matches ({cov}% JD coverage)")
    elif exact > 0:
        parts.append(f"~ {exact} exact keyword matches ({cov}% JD coverage)")
    else:
        parts.append("⚠ No direct skill matches found — relying on semantic inference")

    if score >= 80:
        parts.append("→ Strong fit overall")
    elif score >= 67:
        parts.append("→ Good fit with minor gaps")
    elif score >= 52:
        parts.append("→ Moderate fit — some gaps to address")
    elif score >= 38:
        parts.append("→ Partial fit — significant gaps present")
    else:
        parts.append("→ Weak fit — critical skill gaps")

    return " | ".join(parts)


def _sse(payload: dict) -> str:
    return "data: " + json.dumps(payload) + "\n\n"


# ─── Routes ─────────────────────────────────────────────────────────────────


@router.get("/", tags=["root"])
async def root():
    idx = FRONTEND_DIR / "index.html"
    return FileResponse(str(idx)) if idx.exists() else {"message": "ATS API v3 — see /docs"}


@router.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}


@router.get("/info", tags=["info"])
async def info():
    return get_model_info()


@router.get("/info/wait", tags=["info"])
async def info_wait():
    for _ in range(120):
        d = get_model_info()
        if d.get("backend") != "loading":
            return d
        await asyncio.sleep(1)
    return get_model_info()


@router.post("/analyze", response_model=ATSResult, tags=["analysis"])
async def analyze(
    request: Request,
    resume: UploadFile = File(...),
    job_description: UploadFile = File(None),
    jd_text: str = Form(None),
    domain: str = Form(None),
):
    """Analyze resume against job description — returns full ATSResult."""
    try:
        rb  = await _validate_file(resume, "Resume")
        rfn = resume.filename or "resume.pdf"
        if job_description:
            jd_text = await _parse_jd_upload(job_description)
        elif not jd_text:
            raise HTTPException(422, "Provide job_description file or jd_text field.")
        async def _noop(s, m, p): pass
        return await _run_pipeline(rb, rfn, jd_text, _noop, domain)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analyze failed: {e}", exc_info=True)
        raise HTTPException(500, "Internal server error during analysis")


@router.post("/analyze/stream", tags=["analysis"])
async def analyze_stream(
    request: Request,
    resume: UploadFile = File(...),
    job_description: UploadFile = File(None),
    jd_text: str = Form(None),
    domain: str = Form(None),
):
    """Streaming analysis with SSE progress updates."""
    try:
        rb = await _validate_file(resume, "Resume")
    except HTTPException as exc:
        async def _err():
            yield _sse({"stage": "error", "message": exc.detail})
        return StreamingResponse(_err(), media_type="text/event-stream")

    rfn    = resume.filename or "resume.pdf"
    jd_err = None

    if job_description:
        try:
            jd_text = await _parse_jd_upload(job_description)
        except HTTPException as exc:
            jd_err = exc.detail
    elif not jd_text:
        jd_err = "No job description provided."

    if jd_err:
        msg = jd_err
        async def _err():
            yield _sse({"stage": "error", "message": msg})
        return StreamingResponse(_err(), media_type="text/event-stream")

    queue: asyncio.Queue = asyncio.Queue()

    async def progress(stage, message, pct):
        await queue.put({"stage": stage, "message": message, "pct": pct})

    async def generator():
        task = asyncio.create_task(_run_pipeline(rb, rfn, jd_text, progress, domain))
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.1)
                yield _sse(item)
                if item.get("pct", 0) >= 100:
                    break
            except asyncio.TimeoutError:
                if task.done():
                    while not queue.empty():
                        yield _sse(queue.get_nowait())
                    break
        try:
            result = await task
            yield _sse({"stage": "result", "data": result.model_dump()})
        except Exception as exc:
            yield _sse({"stage": "error", "message": str(exc)})

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/graph/related", tags=["graph"])
async def graph_related(skill: str, top_k: int = 5):
    from backend.graph.skill_graph import DynamicSkillGraph
    g = DynamicSkillGraph()
    g.build([[skill]])
    related = g.top_related(skill.lower(), top_k=top_k)
    return {"skill": skill, "related": [{"skill": s, "weight": round(w, 3)} for s, w in related]}


@router.get("/graph/adjacency", tags=["graph"])
async def graph_adjacency():
    return {"message": "Graph built per-request. Use /analyze for inferred_skills and skill_gap."}

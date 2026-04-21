from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from backend.app.schemas.ats import (
    ATSAnalysisRequest,
    ATSResultResponse,
    BreakdownResponse,
    DocumentParseRequest,
    JDStructureRequest,
    KeywordsResponse,
    LLMAnalysisResponse,
    ParsedDocumentResponse,
    ParsedPairResponse,
    ScoreResponse,
    StructuredJobDescriptionResponse,
    SuggestionsResponse,
)
from backend.services.ats_service import ATSService

router = APIRouter(prefix="/ats", tags=["ats"])
service = ATSService()


@router.post(
    "/evaluate",
    summary="LLM evaluation",
    description="Returns only the LLM evaluation with score, grade, summary, strengths, risks, and interview guidance.",
)
def evaluate_resume(payload: ATSAnalysisRequest) -> dict:
    result = service.get_llm_analysis(
        resume_text=payload.resume_text,
        job_text=payload.job_text,
    )
    return result["evaluation"]


@router.post(
    "/evaluate-files",
    response_model=ATSResultResponse,
    summary="Full evaluation from files",
    description="Multipart file upload to full LLM-guided ATS result. Accepts PDF, DOCX, and TXT with temp file cleanup handled server-side.",
)
async def evaluate_resume_files(
    resume: UploadFile = File(...),
    job: UploadFile = File(...),
) -> dict:
    return await service.evaluate_files(resume=resume, job=job)


@router.post(
    "/parse",
    response_model=ParsedDocumentResponse,
    summary="Parse text",
    description="Parse any text into source path, raw text, detected sections, and metadata.",
)
def parse_text(payload: DocumentParseRequest) -> dict:
    return service.parse_text(
        text=payload.text,
        source_name=payload.source_name,
    )


@router.post(
    "/parse-resume",
    response_model=ParsedDocumentResponse,
    summary="Parse resume text",
    description="Parse text as resume content with source set to resume_text.",
)
def parse_resume_text(payload: DocumentParseRequest) -> dict:
    return service.parse_resume_text(payload.text)


@router.post(
    "/parse-job",
    response_model=ParsedDocumentResponse,
    summary="Parse job text",
    description="Parse text as job description content with source set to job_text.",
)
def parse_job_text(payload: DocumentParseRequest) -> dict:
    return service.parse_job_text(payload.text)


@router.post(
    "/parse-pair",
    response_model=ParsedPairResponse,
    summary="Parse resume and job text",
    description="Parse resume text and job text together and return both parsed documents in one response.",
)
def parse_text_pair(payload: ATSAnalysisRequest) -> dict:
    return service.parse_text_pair(
        resume_text=payload.resume_text,
        job_text=payload.job_text,
    )


@router.post(
    "/parse-file",
    response_model=ParsedDocumentResponse,
    summary="Parse uploaded file",
    description="Upload any supported file and return the parsed document. PDF parsing cascades PyMuPDF to pdfplumber.",
)
async def parse_file(file: UploadFile = File(...)) -> dict:
    return await service.parse_file(file)


@router.post(
    "/parse-resume-file",
    response_model=ParsedDocumentResponse,
    summary="Parse uploaded resume file",
    description="Alias of parse-file with the multipart field named resume.",
)
async def parse_resume_file(resume: UploadFile = File(...)) -> dict:
    return await service.parse_resume_file(resume)


@router.post(
    "/parse-job-file",
    response_model=ParsedDocumentResponse,
    summary="Parse uploaded job file",
    description="Alias of parse-file with the multipart field named job.",
)
async def parse_job_file(job: UploadFile = File(...)) -> dict:
    return await service.parse_job_file(job)


@router.post(
    "/parse-files",
    response_model=ParsedPairResponse,
    summary="Parse resume and job files",
    description="Upload both files and return the parsed pair. Temporary files are cleaned up in a finally block.",
)
async def parse_files(
    resume: UploadFile = File(...),
    job: UploadFile = File(...),
) -> dict:
    return await service.parse_files(resume=resume, job=job)


@router.post(
    "/score",
    response_model=ScoreResponse,
    summary="LLM-guided score",
    description="Returns the final hybrid score and grade using LLM assessment, semantic similarity, and keyword coverage.",
)
def get_score(payload: ATSAnalysisRequest) -> dict:
    return service.get_score(
        resume_text=payload.resume_text,
        job_text=payload.job_text,
    )


@router.post(
    "/breakdown",
    response_model=BreakdownResponse,
    summary="LLM-guided score breakdown",
    description="Returns final score, grade, semantic similarity, keyword coverage, section completeness, and per-section scores.",
)
def get_breakdown(payload: ATSAnalysisRequest) -> dict:
    return service.get_breakdown(
        resume_text=payload.resume_text,
        job_text=payload.job_text,
    )


@router.post(
    "/keywords",
    response_model=KeywordsResponse,
    summary="Keyword analysis",
    description="Runs keyword extraction and fuzzy matching only. Returns coverage, missing keywords, and keyword matches with similarity scores.",
)
def get_keywords(payload: ATSAnalysisRequest) -> dict:
    return service.get_keywords(
        resume_text=payload.resume_text,
        job_text=payload.job_text,
    )


@router.post(
    "/suggestions",
    response_model=SuggestionsResponse,
    summary="LLM suggestions",
    description="Runs the full LLM-guided pipeline and returns the grade-derived score, grade, and section-wise improvement suggestions.",
)
def get_suggestions(payload: ATSAnalysisRequest) -> dict:
    return service.get_suggestions(
        resume_text=payload.resume_text,
        job_text=payload.job_text,
    )


@router.post(
    "/llm-analysis",
    response_model=LLMAnalysisResponse,
    summary="LLM-only analysis",
    description="Returns only the LLM outputs: structured JD, skill assessment, and resume suggestions. No score or breakdown fields are returned.",
)
def get_llm_analysis(payload: ATSAnalysisRequest) -> dict:
    return service.get_llm_analysis(
        resume_text=payload.resume_text,
        job_text=payload.job_text,
    )


@router.post(
    "/structure-jd",
    response_model=StructuredJobDescriptionResponse,
    summary="Structure job description",
    description="Standalone JD extraction with no resume input. Returns required skills, preferred skills, experience years, education level, and responsibilities.",
)
def structure_job_description(payload: JDStructureRequest) -> dict:
    return service.structure_job_description(payload.job_text)

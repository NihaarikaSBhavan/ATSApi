from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import UploadFile

from backend.app.core.settings import get_settings
from atsv4 import ATSConfig, ATSEngine
from atsv4.llm import structure_job_description as llm_structure_job_description
from atsv4.parsers import parse_document, parse_text_document


class ATSService:
    def __init__(self) -> None:
        settings = get_settings()
        config = ATSConfig(
            embedding_model_name=settings.ats_embedding_model,
            llm_model_name=settings.ats_llm_model,
            llm_api_key=settings.llm_api_key,
            llm_base_url=settings.llm_base_url,
        )
        self.engine = ATSEngine(config)
        self.config = config

    def evaluate_text(self, resume_text: str, job_text: str) -> dict:
        result = self.engine.evaluate_texts(
            resume_text=resume_text,
            job_text=job_text,
            use_llm=True,
        )
        return result.to_dict()

    def parse_text(self, text: str, source_name: str = "inline_text") -> dict:
        parsed = parse_text_document(text, source_name=source_name)
        return self._serialize_parsed_document(parsed)

    def parse_resume_text(self, resume_text: str) -> dict:
        parsed = parse_text_document(resume_text, source_name="resume_text")
        return self._serialize_parsed_document(parsed)

    def parse_job_text(self, job_text: str) -> dict:
        parsed = parse_text_document(job_text, source_name="job_text")
        return self._serialize_parsed_document(parsed)

    def parse_text_pair(self, resume_text: str, job_text: str) -> dict:
        return {
            "resume": self.parse_resume_text(resume_text),
            "job": self.parse_job_text(job_text),
        }

    def get_score(self, resume_text: str, job_text: str) -> dict:
        result = self.engine.evaluate_texts(
            resume_text=resume_text,
            job_text=job_text,
            use_llm=True,
        )
        return {
            "score": result.score,
            "grade": result.grade,
        }

    def get_breakdown(self, resume_text: str, job_text: str) -> dict:
        result = self.engine.evaluate_texts(
            resume_text=resume_text,
            job_text=job_text,
            use_llm=True,
        )
        return {
            "score": result.score,
            "grade": result.grade,
            "semantic_similarity": result.semantic_similarity,
            "keyword_coverage": result.keyword_coverage,
            "section_completeness": result.section_completeness,
            "section_scores": result.section_scores,
        }

    def get_keywords(self, resume_text: str, job_text: str) -> dict:
        result = self.engine.evaluate_texts(
            resume_text=resume_text,
            job_text=job_text,
            use_llm=False,
        )
        return {
            "keyword_coverage": result.keyword_coverage,
            "missing_keywords": result.missing_keywords,
            "keyword_matches": [
                {
                    "job_keyword": match.job_keyword,
                    "resume_keyword": match.resume_keyword,
                    "similarity": match.similarity,
                }
                for match in result.keyword_matches
            ],
        }

    def get_suggestions(self, resume_text: str, job_text: str) -> dict:
        result = self.engine.evaluate_texts(
            resume_text=resume_text,
            job_text=job_text,
            use_llm=True,
        )
        return {
            "score": result.score,
            "grade": result.grade,
            "evaluation": (
                result.evaluation.to_dict()
                if result.evaluation is not None
                else None
            ),
            "suggestions": result.suggestions,
        }

    def get_llm_analysis(self, resume_text: str, job_text: str) -> dict:
        result = self.engine.evaluate_texts(
            resume_text=resume_text,
            job_text=job_text,
            use_llm=True,
        )
        return {
            "structured_job_description": (
                result.structured_job_description.to_dict()
                if result.structured_job_description is not None
                else None
            ),
            "evaluation": (
                result.evaluation.to_dict()
                if result.evaluation is not None
                else None
            ),
            "suggestions": result.suggestions,
        }

    async def evaluate_files(self, resume: UploadFile, job: UploadFile) -> dict:
        resume_path = await self._persist_upload(resume)
        job_path = await self._persist_upload(job)
        try:
            result = self.engine.evaluate(
                resume_path=resume_path,
                job_path=job_path,
                use_llm=True,
            )
            return result.to_dict()
        finally:
            for path in (resume_path, job_path):
                if path.exists():
                    os.unlink(path)

    async def parse_file(self, upload: UploadFile) -> dict:
        file_path = await self._persist_upload(upload)
        try:
            parsed = parse_document(file_path)
            return self._serialize_parsed_document(parsed)
        finally:
            if file_path.exists():
                os.unlink(file_path)

    async def parse_resume_file(self, resume: UploadFile) -> dict:
        return await self.parse_file(resume)

    async def parse_job_file(self, job: UploadFile) -> dict:
        return await self.parse_file(job)

    async def parse_files(self, resume: UploadFile, job: UploadFile) -> dict:
        resume_path = await self._persist_upload(resume)
        job_path = await self._persist_upload(job)
        try:
            return {
                "resume": self._serialize_parsed_document(parse_document(resume_path)),
                "job": self._serialize_parsed_document(parse_document(job_path)),
            }
        finally:
            for path in (resume_path, job_path):
                if path.exists():
                    os.unlink(path)

    def structure_job_description(self, job_text: str) -> dict:
        return llm_structure_job_description(job_text, self.config).to_dict()

    async def _persist_upload(self, upload: UploadFile) -> Path:
        suffix = Path(upload.filename or "upload.txt").suffix or ".txt"
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await upload.read()
            temp_file.write(content)
            return Path(temp_file.name)

    def _serialize_parsed_document(self, parsed) -> dict:
        return {
            "source_path": parsed.source_path,
            "raw_text": parsed.raw_text,
            "sections": parsed.sections,
            "metadata": parsed.metadata,
        }

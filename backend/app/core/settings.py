from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(slots=True)
class Settings:
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    frontend_origin: str = os.getenv("FRONTEND_ORIGIN", "http://127.0.0.1:5500")
    ats_embedding_model: str = os.getenv("ATS_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    ats_llm_model: str = os.getenv("AI_MODEL") or os.getenv("ATS_LLM_MODEL", "gpt-4o-mini")
    llm_api_key: str | None = (
        os.getenv("LLM_API_KEY")
        or os.getenv("GROK_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    llm_base_url: str | None = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")


def get_settings() -> Settings:
    return Settings()

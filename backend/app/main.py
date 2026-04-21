from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.ats import router as ats_router
from backend.app.api.routes.health import router as health_router
from backend.app.core.settings import get_settings

settings = get_settings()
allowed_origins = [
    settings.frontend_origin,
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "null",
]
local_dev_origin_regex = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"

app = FastAPI(
    title="ATSv4 API",
    version="0.1.0",
    description="Backend API for LLM-guided ATS evaluation grounded by semantic and keyword evidence.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=local_dev_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api/v1")
app.include_router(ats_router, prefix="/api/v1")

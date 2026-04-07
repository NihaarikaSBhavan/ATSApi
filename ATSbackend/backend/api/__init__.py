"""ATS API application factory."""
import asyncio
import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.extraction.skill_extractor import preload_async
from backend.embeddings.embedder import preload as preload_embedder

from .config import get_frontend_dir, setup_logging
from .middleware import setup_cors, setup_request_logging, setup_rate_limiting
from .routes import router

logger = setup_logging()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="AI ATS Engine", version="3.0.0")

    setup_rate_limiting(app)
    setup_cors(app)
    setup_request_logging(app)

    frontend_dir = get_frontend_dir()
    if frontend_dir.exists() and frontend_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(frontend_dir.resolve())), name="static")
        logger.info(f"Frontend mounted from {frontend_dir.resolve()}")
    else:
        logger.warning(f"Frontend not found at {frontend_dir.resolve()}")

    app.include_router(router)

    @app.on_event("startup")
    async def startup_event():
        logger.info("ATS Engine starting up...")
        # Load models in background so port binds immediately
        asyncio.create_task(_load_models())

    async def _load_models():
        logger.info("Pre-loading SentenceTransformer embedder...")
        await asyncio.to_thread(preload_embedder)
        logger.info("Embedder ready.")
        await asyncio.to_thread(preload_async)
        logger.info("Skill extractor ready.")
        logger.info("ATS Engine startup complete.")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("ATS Engine shutting down.")

    return app


app = create_app()
__all__ = ["app", "create_app"]

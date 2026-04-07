"""Middleware and rate limiting configuration."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address

from .config import get_cors_origins

logger = logging.getLogger(__name__)


def setup_rate_limiting(app: FastAPI) -> Limiter:
    """Configure rate limiting for the API."""
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    return limiter


def setup_cors(app: FastAPI):
    """Configure CORS middleware."""
    origins = get_cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
        allow_credentials=False,
    )
    logger.info(f"CORS configured for origins: {origins}")


def setup_request_logging(app: FastAPI):
    """Setup HTTP request/response logging middleware."""
    @app.middleware("http")
    async def log_requests(request, call_next):
        """Log all HTTP requests with method, path, and response status."""
        client_host = request.client.host if request.client else "unknown"
        logger.info(f"{request.method} {request.url.path} from {client_host}")
        
        try:
            response = await call_next(request)
            logger.debug(f"{request.method} {request.url.path} -> {response.status_code}")
            return response
        except Exception as exc:
            logger.error(f"{request.method} {request.url.path} raised exception: {exc}", exc_info=True)
            raise

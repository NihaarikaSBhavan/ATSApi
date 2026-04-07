"""Configuration and logging setup for ATS API."""
import logging
import os
from pathlib import Path


def setup_logging():
    """Configure structured logging with environment-based level."""
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


def get_cors_origins():
    """Get CORS allowed origins from environment."""
    origins_str = os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:8000"
    )
    return [origin.strip() for origin in origins_str.split(",")]


def get_max_upload_size():
    """Get maximum upload file size in bytes."""
    return int(os.environ.get("MAX_UPLOAD_SIZE_BYTES", "10485760"))  # 10 MB default


def get_frontend_dir():
    """Get frontend directory path."""
    return Path(__file__).parent.parent.parent / "frontend"

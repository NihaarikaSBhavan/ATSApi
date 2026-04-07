
"""ATS Engine entry point."""
import os
import uvicorn

from backend.api import app

if __name__ == "__main__":
    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # Disable auto-reload for stability
    )


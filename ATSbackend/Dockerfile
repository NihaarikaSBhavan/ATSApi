FROM python:3.12-slim

# System dependencies for pdfplumber and document parsing
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model (small, always needed)
RUN python -m spacy download en_core_web_sm

# Copy source code
COPY . .

# Pre-train the GBM model at build time so first request is instant
RUN python -c "import sys; sys.path.insert(0,''); from backend.model.trainer import train_model; train_model(); print('Model pre-trained')"

EXPOSE 8000

# Single worker — ML models are singletons and not thread-safe across workers
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]

# AI ATS Engine

**Automated resume screening and skill matching system using AI.**

An intelligent Applicant Tracking System (ATS) that analyzes resumes against job descriptions using NLP, skill extraction, and machine learning to provide data-driven hiring decisions.

## Features

- **Resume Analysis** — Extract skills and match against job descriptions
- **Semantic Matching** — Similarity scoring using transformer embeddings
- **Skill Inference** — Infer related skills via dynamic skill graph
- **Real-time Streaming** — Server-Sent Events for live progress updates
- **Production Ready** — Rate limiting, CORS, structured logging, input validation
- **Explainable Scores** — Feature breakdown showing match factors
- **Thread-Safe** — Robust model caching and concurrent request handling

## Quick Start

### Prerequisites

- Python 3.11+
- pip or conda

### Installation

```bash
# Clone repository
git clone <repo-url>
cd ATSProd

# Install dependencies
pip install -r requirements.txt

# Create environment config (optional)
cp .env.example .env
```

### Run the Server

```bash
# Development mode (auto-reload)
python main.py

# Or use uvicorn directly
uvicorn backend.api:app --reload --port 8000
```

Server starts at: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

## API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### Health Check
```bash
GET /health
# Returns: {"status": "ok"}
```

#### Model Status
```bash
GET /info
# Returns: {"backend": "loaded|loading", "model": "ready", ...}

GET /info/wait
# Blocks up to 120s until model is loaded
```

#### Analyze Resume (Request-Response)
```bash
POST /analyze
Content-Type: multipart/form-data

resume: <PDF|DOCX|TXT file>
job_description: <PDF|DOCX|TXT file> OR jd_text: "Job description text"

# Response
{
  "score": 78.5,
  "label": "Good Match",
  "resume_skills": ["Python", "FastAPI", "Machine Learning"],
  "jd_skills": ["Python", "FastAPI", "Docker", "Kubernetes"],
  "matched_skills": ["Python", "FastAPI"],
  "missing_skills": ["Docker", "Kubernetes"],
  "inferred_skills": ["REST API", "Web Development"],
  "skill_gap": [...],
  "features": {
    "global_similarity": 0.75,
    "skill_embedding_soft": 0.80,
    "keyword_overlap": 0.35,
    "graph_soft_match": 0.78,
    "inferred_coverage": 0.65,
    "weighted_composite": 0.736
  }
}
```

#### Analyze Resume (Streaming)
```bash
POST /analyze/stream
Content-Type: multipart/form-data

resume: <file>
job_description: <file> OR jd_text: "..."

# Response: Server-Sent Events stream
data: {"stage": "parse_resume", "message": "Parsing resume...", "pct": 10}
data: {"stage": "extract_resume", "message": "Extracting skills...", "pct": 25}
data: {"stage": "extract_jd", "message": "Extracting JD skills...", "pct": 40}
data: {"stage": "skill_graph", "message": "Building graph...", "pct": 55}
data: {"stage": "features", "message": "Engineering features...", "pct": 70}
data: {"stage": "predict", "message": "Running model...", "pct": 88}
data: {"stage": "done", "message": "Complete - Good Match (78.5)", "pct": 100}
data: {"stage": "result", "data": {...full ATSResult...}}
```

#### Skill Graph
```bash
GET /graph/related?skill=python&top_k=5
# Returns related skills and similarity weights

GET /graph/adjacency
# Returns info about skill graph
```

### Rate Limiting

Analysis endpoints are rate-limited to **5 requests per minute** per IP address.

### File Upload Constraints

- Maximum file size: **10 MB** (configurable)
- Supported formats: PDF, DOCX, TXT
- Empty files are rejected

## Configuration

Configuration is managed via environment variables in `.env`:

```bash
# API Security
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000

# Model Loading
FORCE_REGEX=0                          # Set to 1 to skip SLM, use regex only
MIN_FREE_MB=600                        # Minimum free memory before loading model
SLM_LOAD_TIMEOUT=90                    # Timeout for model loading (seconds)

# File Upload
MAX_UPLOAD_SIZE_BYTES=10485760         # 10 MB

# Server
PORT=8000
LOG_LEVEL=INFO                         # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

See [.env.example](.env.example) for all options.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                     main.py                          │
│         (Entry point - runs uvicorn)                │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│            backend/api/__init__.py                   │
│         (App factory - creates FastAPI app)         │
├─────────────────────────────────────────────────────┤
│  ├─ config.py        (Environment setup)            │
│  ├─ middleware.py    (CORS, logging, rate limit)    │
│  ├─ routes.py        (All endpoints)                │
│  └─ models.py        (Pydantic response models)     │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│              ML Pipeline (backend/)                  │
├─────────────────────────────────────────────────────┤
│  ├─ parser/          (PDF/DOCX/TXT parsing)        │
│  ├─ extraction/      (Skill extraction)             │
│  ├─ embeddings/      (Semantic embeddings)          │
│  ├─ graph/           (Skill graph inference)        │
│  ├─ features/        (Feature engineering)          │
│  ├─ scoring/         (ML prediction)                │
│  └─ model/           (Model training/caching)       │
└─────────────────────────────────────────────────────┘
```

### Key Components

**API Module** (`backend/api/`)
- Modular, testable architecture
- Separated concerns: routing, middleware, config
- Server-Sent Events streaming
- Rate limiting via slowapi
- Structured logging

**ML Pipeline** (`backend/`)
- **Parser** — Extracts text from documents
- **Extractor** — 3-tier skill detection (SLM → spaCy → regex)
- **Embeddings** — Sentence transformer models
- **Graph** — Dynamic skill relationship inference
- **Features** — 6-dimensional feature vector
- **Scorer** — GradientBoosting ML model
- **Model** — Thread-safe caching and serialization

## Development

### Project Structure

```
ATSProd/
├── main.py                          # Entry point
├── requirements.txt                 # Dependenices
├── .env                            # Local configuration
├── .env.example                    # Configuration template
├── Dockerfile                      # Docker image
├── docker-compose.yml              # Multi-container setup
├── DEPLOY.md                       # Deployment guide
├── test_startup.py                 # Startup verification
├── test_api_refactor.py           # API module tests
│
├── backend/
│   ├── api/                        # FastAPI application
│   │   ├── __init__.py            # App factory
│   │   ├── config.py              # Configuration
│   │   ├── middleware.py          # Middleware setup
│   │   ├── models.py              # Pydantic models
│   │   └── routes.py              # API endpoints
│   │
│   ├── parser/                    # Document parsing
│   ├── extraction/                # Skill extraction
│   ├── embeddings/                # Embeddings
│   ├── graph/                     # Skill graph
│   ├── features/                  # Feature engineering
│   ├── scoring/                   # ML prediction
│   └── model/                     # Model training
│
├── frontend/                       # Static HTML
├── nginx/                         # Reverse proxy config
└── DEPLOY.md                      # Deployment guide
```

### Running Tests

```bash
# Startup verification
python test_startup.py

# API module tests
python test_api_refactor.py
```

### Development Workflow

1. **Make changes** to any backend module
2. **Save file** — server auto-reloads
3. **Test endpoint** — http://localhost:8000/docs
4. **Check logs** — structured logging output

### Adding New Endpoints

```python
# In backend/api/routes.py
@router.post("/new-endpoint", tags=["custom"])
async def new_endpoint(request: Request, param: str):
    """Your endpoint description."""
    return {"result": "value"}
```

Routes are auto-included in `backend/api/__init__.py`:
```python
app.include_router(router)
```

## Deployment

### Docker

```bash
# Build image
docker build -t ats-engine .

# Run container
docker run -p 8000:8000 ats-engine
```

### Docker Compose

```bash
# Start all services (API + nginx)
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Cloud Deployment

See [DEPLOY.md](DEPLOY.md) for detailed instructions:
- **Render** 
https://atsapi.onrender.com

## Performance

### Benchmarks (on test data)

- **Resume parsing**: ~200ms

### Scoring Behavior (practical production variant)

- Model uses **XGBoost with 12 feature vector** (semantic, embedding, keyword, graph, inference, composite).
- When `weighted_composite >= 0.75` and `inferred_coverage >= 0.9`, score is boosted via:
  - `practical_score = 50 + weighted_composite * 35`
  - Final score is `max(model_score, practical_score)` (with 0-100 clamp).
- When `weighted_composite >= 0.70` and `inferred_coverage >= 0.8`, score is boosted via:
  - `practical_score = 45 + weighted_composite * 33`
  - Final score is `max(model_score, practical_score)`.

### Why this is production-safe

1. avoids low scores for candidates with strong inferred skill coverage
2. preserves trained model behavior for low/uncertain cases
3. ensures conservative bias for weak matches and liberal bias for high inference
4. supports auditability via `raw_model_score` + `practical_score` logging

### Multi-Domain Skill Graph Generalization

The ATS engine now supports **10+ industry domains** with domain-specific skill relationships and inference rules:

#### Tier 1: Domain-Configurable Skill Dependencies

Each domain has curated skill dependency graphs that model domain-specific relationships:

- **Tech** (27 dependencies): `python→numpy→deep-learning→tensorflow→keras`
- **Healthcare** (20 dependencies): `patient-care→medication→iv-therapy`, `surgery→anatomy→physiology`
- **Finance** (19 dependencies): `portfolio→risk-management`, `derivatives→financial-modeling`
- **Manufacturing** (18 dependencies): `supply-chain→inventory→lean-manufacturing→six-sigma`
- **Sales/Marketing** (varied): `account-management→crm→salesforce`, `digital-marketing→conversion-optimization`
- **Logistics** (varied): `warehouse-management→inventory-control`, `procurement→vendor-management`
- **HR, Education, Construction, Legal** — Full domain coverage with industry-standard skill hierarchies

**Usage:**
```python
from backend.graph.skill_graph import SkillGraph

# Auto-detect domain from JD, or specify explicitly:
graph = SkillGraph(domain='healthcare')
# or
graph = SkillGraph(domain='finance')
```

#### Tier 2: Auto-Detection from Job Description

The API automatically detects domain from job description text using keyword matching and job title analysis:

```python
from backend.graph.skill_graph import detect_domain

jd = "Senior Nurse with EHR expertise and HIPAA compliance..."
domain = detect_domain(jd)  # Returns: 'healthcare'
```

**How it works:**
1. Scans JD for domain-specific keywords (e.g., "HIPAA" → healthcare, "React" → tech)
2. Weights job titles higher (e.g., "Registered Nurse" → 3x score boost)
3. Returns domain with highest keyword+title score
4. Falls back to 'tech' if ambiguous

#### Tier 3: Embedding-Based Fallback

When hardcoded dependencies don't exist, the system falls back to pure semantic similarity:
- Soft matching via embeddings (all-MiniLM-L6-v2 transformer)
- Co-occurrence analysis from resume+JD corpus
- Fuzzy string matching for near-misses

**Result:** Multi-skill inference works across **all domains**, not just tech.

#### Domain Response Field

API response now includes detected domain for transparency:
```json
{
  "score": 79.9,
  "label": "Excellent Match",
  "domain": "healthcare",
  "matched_skills": ["patient care", "medication management", "iv therapy"],
  "inferred_skills": ["clinical assessment", "patient psychology"]
}
```

**Implementation Details:**
- `backend/graph/skill_graph.py`: TECH_DEPS, HEALTHCARE_DEPS, FINANCE_DEPS, etc. + `detect_domain()` function
- `backend/api/routes.py`: Auto-detects domain, passes to SkillGraph during pipeline
- `backend/api/models.py`: ATSResult includes `domain` field
- Thread-safe: Domain detection runs in thread pool, doesn't block async handler

- **Skill extraction**: ~150ms (SLM) or ~10ms (regex)
- **Full pipeline**: ~1.5s total (with model loaded)
- **Concurrent requests**: 5 req/min rate limit (configurable)
- **Memory**: ~2GB for model + embeddings

### Optimizations

- Thread-safe model caching (load once, reuse)
- Lazy loading of SentenceTransformer
- Async I/O for all blocking operations
- MinMaxScaler for feature normalization
- joblib serialization (safer than pickle)

## Troubleshooting

### Model Takes Too Long to Load

```bash
# Use regex-only mode (faster, less accurate)
FORCE_REGEX=1 python main.py
```

### High Memory Usage

- Increase `MIN_FREE_MB` to skip SLM if memory constrained
- Use `FORCE_REGEX=1` to avoid loading neural models

### Import Errors

Make sure you're using the right Python environment:
```bash
# NOT this (system Python):
C:\Users\Admin\AppData\Local\Microsoft\WindowsApps\python3.11.exe

# Use this (project Python):
cd ATSProd
python main.py
```

### Rate Limit Exceeded

The API limits `/analyze` endpoints to **5 requests per minute**. Wait 60 seconds before retrying.

## Security

### Features

- CORS restricted to known origins
- File upload validation (size + emptiness checks)
- Input sanitization via Pydantic models
- Rate limiting to prevent abuse
- Error handling (sensitive details logged, generic responses returned)
- Safe model serialization (joblib instead of pickle)
- Thread-safe concurrent access

### Best Practices

1. **Production**: Set `ALLOWED_ORIGINS` to your domain only
2. **Logging**: Use `LOG_LEVEL=INFO` in production (not DEBUG)
3. **API Keys**: Implement authentication before production use
4. **HTTPS**: Deploy behind reverse proxy with TLS

## Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes and test
4. Commit: `git commit -m "Add feature"`
5. Push: `git push origin feature/my-feature`
6. Open Pull Request

## License

MIT License — See LICENSE file for details

## Support

- **Issues**: GitHub Issues
- **Documentation**: See DEPLOY.md for deployment details
- **API Docs**: http://localhost:8000/docs (live server required)


---

**Version**: 2.0.0  
**Last Updated**: March 26, 2026  
**Status**: Production Ready

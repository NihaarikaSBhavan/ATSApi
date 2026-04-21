# ATSv4

ATSv4 is an LLM-guided applicant tracking scorer grounded by two evidence tracks, with a separated backend, frontend, and service layer:

- Semantic similarity with sentence-transformer embeddings
- Keyword analysis with phrase extraction plus fuzzy matching

The evaluation path structures job descriptions, evaluates matched/missed/inferred skills, estimates hiring potential for an admin portal, assigns a candidate grade, and blends that grade with semantic similarity and keyword coverage to produce the final numerical score.

## Features

- PDF and DOCX resume ingestion with graceful fallbacks
- Resume section detection with fuzzy header matching
- Section-level semantic scoring to avoid full-document truncation issues
- JD keyword extraction and RapidFuzz matching
- Hybrid LLM scoring with semantic and keyword evidence
- OpenAI-compatible JD structuring and resume feedback generation
- FastAPI backend with API routes
- Service layer that wraps the ATS engine
- Lightweight frontend starter page
- CLI output in JSON for easy API or UI integration

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Start the backend:

```bash
uvicorn backend.app.main:app --reload
```

Run a CLI evaluation:

```bash
atsv4 evaluate --resume path\to\resume.pdf --job path\to\job_description.txt
```

Open the starter frontend by serving the `frontend` folder with any static server.

## Project Structure

```text
backend/
  app/
    api/routes/
    core/
    schemas/
  services/
frontend/
src/atsv4/
tests/
```

## API Routes

Base path: `/api/v1`

Health:

- `GET /health` -> liveness check returning `{"status":"ok"}`

Full evaluation:

- `POST /ats/evaluate` -> `{ resume_text, job_text }` to full LLM-guided ATS result
- `POST /ats/evaluate-files` -> multipart `resume`, `job` to full LLM-guided ATS result

Scoring:

- `POST /ats/score` -> final hybrid LLM-guided `score` and `grade`
- `POST /ats/breakdown` -> score plus semantic, keyword, completeness, and per-section breakdown

Keyword analysis:

- `POST /ats/keywords` -> keyword coverage, missing keywords, and fuzzy matches

LLM endpoints:

- `POST /ats/structure-jd` -> standalone structured job description extraction
- `POST /ats/suggestions` -> LLM-derived score, grade, and section-wise improvement suggestions
- `POST /ats/llm-analysis` -> structured JD, LLM skill assessment, and suggestions only

Parsing utilities:

- `POST /ats/parse` -> parse arbitrary text with optional `source_name`
- `POST /ats/parse-resume` -> parse text as `resume_text`
- `POST /ats/parse-job` -> parse text as `job_text`
- `POST /ats/parse-pair` -> parse resume and JD together
- `POST /ats/parse-file` -> multipart `file` to parsed document
- `POST /ats/parse-resume-file` -> multipart `resume` to parsed document
- `POST /ats/parse-job-file` -> multipart `job` to parsed document
- `POST /ats/parse-files` -> multipart `resume`, `job` to parsed pair

Text and file evaluation routes run with the LLM-enabled analysis path.

## Architecture

### 1. Ingestion

- `PyMuPDF` for primary PDF extraction
- `pdfplumber` as a fallback for difficult layouts
- `python-docx` for DOCX files

All documents are normalized into a shared internal structure with raw text, metadata, and best-effort section splits.

### 2. Semantic Similarity

The system embeds resume sections and the full job description with `all-MiniLM-L6-v2` by default. Per-section scores are weighted into a semantic fit score.

### 3. Keyword Matching

Important phrases are extracted from the job description and resume, then matched with `RapidFuzz` to tolerate wording variation.

### 4. Evidence Factors

```text
Evidence score =
  0.45 * semantic_similarity
  + 0.35 * keyword_coverage
  + 0.20 * section_completeness
```

The semantic and keyword factors are retained as evidence for the LLM and as numerical inputs to the final score.

### 5. LLM-Guided Evaluation

When enabled, the LLM path performs three tasks:

- Structuring the raw job description into JSON
- Structuring the resume signal into JSON and comparing it against the JD
- Producing matched skills, missed skills, inferred relevant skills, hiring potential, hire recommendation, grade, rationale, and suggestions
- Highlighting strengths, risks, and interview focus areas for admin/recruiter review

The server converts the LLM grade to a base numerical score:

```text
A -> 92
B -> 78
C -> 62
D -> 45
E -> 25
```

The final score is a hybrid of LLM judgment and evidence factors:

```text
final_score =
  0.70 * llm_grade_score
  + 0.20 * semantic_similarity_score
  + 0.10 * keyword_coverage_score
```

The final `grade` is derived from `final_score`, while the original LLM grade remains visible in `llm_assessment.grade`.

## Output Shape

The CLI and API return JSON with:

- overall score and grade
- section-level semantic scores
- keyword match details
- completeness metrics
- structured JD data when LLM parsing is enabled
- LLM candidate assessment with matched, missing, and inferred relevant skills
- admin-facing hiring potential, hire recommendation, strengths, risks, and interview focus areas
- section-wise suggestions when LLM feedback is enabled

## Environment

Set values in `.env`:

- `LLM_API_KEY`
- `LLM_BASE_URL`
- `ATS_EMBEDDING_MODEL`
- `ATS_LLM_MODEL`
- `API_HOST`
- `API_PORT`
- `FRONTEND_ORIGIN`

If you are using an OpenAI-compatible provider, set:

```env
LLM_API_KEY=your_key
LLM_BASE_URL=your_provider_base_url
AI_MODEL=your_model_name
```

The app also accepts `GROK_API_KEY` and `GROQ_API_KEY` directly.

If you use `AI_MODEL=llama-3.3-70b-versatile`, that is typically a Groq model identifier rather than a Grok one.

## Notes

- Scanned PDFs are not OCR'd in this scaffold. Add an OCR branch if you expect image-only resumes.
- Multi-column PDFs are partially mitigated through fallbacks, but truly production-grade parsing should include layout-aware testing on Canva-style resumes.

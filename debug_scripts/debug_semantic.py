"""
Debug 1: Check why semantic similarity is low (0.4931)
Analyzing section-by-section scoring and embedding alignment.
"""
import logging
from dotenv import load_dotenv
from atsv4 import ATSConfig, ATSEngine
from atsv4.semantic import score_semantic_similarity
from atsv4.parsers import parse_text_document

load_dotenv()
logging.basicConfig(level=logging.WARNING)

config = ATSConfig()
engine = ATSEngine(config)

resume_text = """John Doe
Software Engineer

EXPERIENCE
- Senior Python Developer at TechCorp (2020-Present)
  - Developed web applications using Django and FastAPI
  - Built REST APIs with proper documentation
  - Implemented machine learning models for data analysis
  - Led a team of 3 junior developers

- Junior Developer at StartupXYZ (2018-2020)
  - Built mobile apps with React Native
  - Worked with Node.js and Express
  - Deployed applications to AWS

SKILLS
- Python, Django, FastAPI
- JavaScript, React, Node.js
- Machine Learning, TensorFlow, Scikit-learn
- Docker, Kubernetes, AWS
- PostgreSQL, MongoDB

EDUCATION
- BS Computer Science, University of Tech, 2018"""

job_text = """Senior Python Developer

We are looking for an experienced Python developer to join our team.

Requirements:
- 3+ years of Python development experience
- Experience with Django or FastAPI
- Knowledge of REST API design
- Experience with cloud platforms (AWS, GCP, or Azure)
- Familiarity with containerization (Docker)
- Bonus: Machine learning experience

Responsibilities:
- Develop and maintain web applications
- Design and implement APIs
- Work with cross-functional teams
- Deploy applications to production
- Mentor junior developers

Skills Required:
- Python (expert level)
- Django or FastAPI
- REST APIs
- AWS or similar cloud platform
- Docker
- PostgreSQL
- Git"""

# Parse documents
resume_doc = parse_text_document(resume_text, source_name="resume")
job_doc = parse_text_document(job_text, source_name="job")

print("=" * 70)
print("DEBUG 1: SEMANTIC SIMILARITY ANALYSIS")
print("=" * 70)
print()

print("Resume sections detected:")
for section_name, content in resume_doc.sections.items():
    if content:
        print(f"  {section_name}: {len(content)} characters, {len(content.split())} words")
print()

print("Job description (no sections in JD):")
print(f"  Raw text: {len(job_doc.raw_text)} characters, {len(job_doc.raw_text.split())} words")
print()

# Score semantic similarity
section_scores, overall_similarity = score_semantic_similarity(
    resume_doc.sections,
    job_doc.raw_text,
    config,
)

print("Semantic Similarity Scores (by section):")
print(f"  Config weights: {config.semantic_section_weights}")
print()

total_weighted = 0
for section_name, weight in config.semantic_section_weights.items():
    score = section_scores.get(section_name, 0)
    contribution = score * weight
    total_weighted += contribution
    status = "✓" if score > 0.5 else "✗" if score == 0 else "⚠"
    print(f"  {status} {section_name:12} score={score:.4f}  weight={weight}  contribution={contribution:.4f}")

print()
print(f"Overall Semantic Similarity: {overall_similarity:.4f}")
print(f"Weighted total (should match): {total_weighted:.4f}")
print()

# Analyze why it might be low
print("Analysis:")
print(f"  - Resume has broad tech skills (React, Node, ML, etc.)")
print(f"  - Job is narrowly focused on Python backend")
print(f"  - Semantic model penalizes non-matching skills")
print(f"  - Some resume content (React, Node) doesn't align with Python JD")
print()

# Section weight recommendations
print("Recommendations:")
print(f"  Current: experience={config.semantic_section_weights['experience']}")
print(f"  Consider: increasing experience weight to 0.50-0.55")
print(f"  Because: 'Senior Python Developer' and 'Built REST APIs' are strong signals")
print()

resume_experience = resume_doc.sections.get("experience", "")
print("Resume Experience Section Preview:")
print(resume_experience[:300] if resume_experience else "  [No experience section found]")

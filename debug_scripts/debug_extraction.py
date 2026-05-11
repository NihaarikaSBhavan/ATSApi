"""
Debug 3: Inspect what Groq skill extraction actually extracts
Shows raw skill graph data before matching
"""
import logging
import json
from dotenv import load_dotenv
from atsv4 import ATSConfig
from atsv4.skill_graph_matcher import _extract_raw

load_dotenv()
logging.basicConfig(level=logging.WARNING)

config = ATSConfig()

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

print("=" * 70)
print("DEBUG 3: SKILL GRAPH EXTRACTION (Raw Groq Output)")
print("=" * 70)
print()

print("Extracting from RESUME...")
resume_skills = _extract_raw(resume_text, "resume", config)
print(f"Extracted {len(resume_skills)} skills from resume:")
for skill in resume_skills:
    relations = skill.get("relations", [])
    rel_str = f" (has {len(relations)} relations)" if relations else ""
    print(f"  - {skill.get('name')} [{skill.get('category', '?')}] ({skill.get('level', '?')}){rel_str}")
print()

print("Extracting from JOB DESCRIPTION...")
job_skills = _extract_raw(job_text, "job_description", config)
print(f"Extracted {len(job_skills)} skills from job description:")
for skill in job_skills:
    relations = skill.get("relations", [])
    rel_str = f" (has {len(relations)} relations)" if relations else ""
    print(f"  - {skill.get('name')} [{skill.get('category', '?')}] ({skill.get('level', '?')}){rel_str}")
print()

# Check for REST API specifically
print("Searching for 'REST' in extracted skills...")
resume_with_rest = [s for s in resume_skills if 'REST' in s.get('name', '').upper() or 'API' in s.get('name', '').upper()]
job_with_rest = [s for s in job_skills if 'REST' in s.get('name', '').upper() or 'API' in s.get('name', '').upper()]
print(f"  Resume: {len(resume_with_rest)} matches")
for s in resume_with_rest:
    print(f"    - {s.get('name')}")
print(f"  Job: {len(job_with_rest)} matches")
for s in job_with_rest:
    print(f"    - {s.get('name')}")
print()

# Print first skill with full details to see structure
if resume_skills:
    print("Sample skill structure (resume, first skill):")
    print(json.dumps(resume_skills[0], indent=2))

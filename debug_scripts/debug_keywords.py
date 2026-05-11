"""
Debug 2: Check full keyword matching list
Shows all extracted keywords and matches, not just top 5
"""
import logging
from dotenv import load_dotenv
from atsv4 import ATSConfig, ATSEngine

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

print("=" * 70)
print("DEBUG 2: FULL KEYWORD EXTRACTION AND MATCHING")
print("=" * 70)
print()

result = engine.evaluate_texts(resume_text, job_text, use_llm=False)

print(f"All Job Keywords Extracted ({len(result.keyword_matches) + len(result.missing_keywords)} total):")
print()

# Sort matches by job keyword name
matches_sorted = sorted(result.keyword_matches, key=lambda x: x.job_keyword)
missing_sorted = sorted(result.missing_keywords)

print(f"MATCHED ({len(matches_sorted)}):")
for i, match in enumerate(matches_sorted, 1):
    print(f"  {i}. {match.job_keyword:30} ← {match.resume_keyword:30} ({match.similarity:6.1f}%)")
print()

print(f"MISSING ({len(missing_sorted)}):")
for i, kw in enumerate(missing_sorted, 1):
    print(f"  {i}. {kw}")
print()

# Analysis
print("Analysis:")
print(f"  Total job keywords identified: {len(matches_sorted) + len(missing_sorted)}")
print(f"  Coverage: {len(matches_sorted)}/{len(matches_sorted) + len(missing_sorted)} = {100*len(matches_sorted)/(len(matches_sorted) + len(missing_sorted)):.1f}%")
print()

# What should be matched but isn't?
important_missing = ["REST API", "PostgreSQL", "AWS", "Cloud"]
found_missing = [m for m in missing_sorted if any(k in m for k in important_missing)]
if found_missing:
    print("⚠ Important keywords marked as MISSING:")
    for kw in found_missing:
        print(f"    - {kw}")
    print()
    print("  Reason: Skill graph didn't match them, or fuzzy match fell below threshold")

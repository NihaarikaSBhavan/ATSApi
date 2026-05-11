"""
Debug 5: FULL PIPELINE TRACE
Trace exactly what happens with REST APIs through the entire pipeline
"""
from dotenv import load_dotenv
from atsv4 import ATSConfig
from atsv4.skill_graph_matcher import extract_and_match

load_dotenv()

config = ATSConfig()

resume_text = """John Doe
Software Engineer

EXPERIENCE
- Senior Python Developer at TechCorp (2020-Present)
  - Built REST APIs with proper documentation

SKILLS
- Python, Django, FastAPI
- Docker, AWS
- PostgreSQL

EDUCATION
- BS Computer Science, University of Tech, 2018"""

job_text = """Senior Python Developer

Requirements:
- Python development 
- REST API design
- Docker
- PostgreSQL

Skills Required:
- Python
- Django
- REST APIs
- AWS"""

print("=" * 70)
print("DEBUG 5: FULL PIPELINE TRACE")
print("=" * 70)
print()

# Call the full matcher
keyword_matches, missing_keywords, coverage, job_kws, resume_kws, context = extract_and_match(
    job_text, resume_text, config
)

print("Job Keywords Extracted (from skill graph):")
for kw in job_kws:
    print(f"  - {kw.phrase} (score: {kw.score:.2f})")
print()

print("Resume Keywords Extracted (from skill graph):")
for kw in resume_kws:
    print(f"  - {kw.phrase} (score: {kw.score:.2f})")
print()

print(f"Coverage: {coverage:.1%}")
print()

print(f"Matched ({len(keyword_matches)}):")
for m in keyword_matches:
    print(f"  - {m.job_keyword:30} <- {m.resume_keyword:30} ({m.similarity:6.1f}%)")
print()

print(f"Missing ({len(missing_keywords)}):")
for m in missing_keywords:
    print(f"  - {m}")
print()

# Look specifically for REST
print("=" * 70)
print("SEARCHING FOR 'REST' SPECIFICALLY")
print("=" * 70)
job_rest = [k for k in job_kws if 'REST' in k.phrase.upper()]
resume_rest = [k for k in resume_kws if 'REST' in k.phrase.upper()]
match_rest = [m for m in keyword_matches if 'REST' in m.job_keyword.upper()]
missing_rest = [m for m in missing_keywords if 'REST' in m.upper()]

print(f"Job keywords with 'REST': {len(job_rest)}")
for k in job_rest:
    print(f"  - {k.phrase}")
print()

print(f"Resume keywords with 'REST': {len(resume_rest)}")
for k in resume_rest:
    print(f"  - {k.phrase}")
print()

print(f"Matches with 'REST': {len(match_rest)}")
for m in match_rest:
    print(f"  - {m.job_keyword}")
print()

print(f"Missing with 'REST': {len(missing_rest)}")
for m in missing_rest:
    print(f"  - {m}")

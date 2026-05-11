"""
Test script to verify skill graph matcher is being used as the main engine.
"""
import logging
from dotenv import load_dotenv
from atsv4 import ATSConfig, ATSEngine

# Load environment variables from .env
load_dotenv()

# Enable logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# Create config
config = ATSConfig()
print("Configuration:")
print(f"  use_skill_graph: {config.use_skill_graph}")
print(f"  groq_api_key: {'SET' if config.groq_api_key else 'NOT SET'}")
print(f"  semantic_weight: {config.semantic_weight}")
print(f"  keyword_weight: {config.keyword_weight}")
print(f"  completeness_weight: {config.completeness_weight}")
print()

# Create engine
engine = ATSEngine(config)
print("✓ ATSEngine created successfully")
print()

# Test text evaluation
resume_text = """
John Doe
Senior Python Developer

EXPERIENCE
- Built REST APIs with Django and FastAPI
- Deployed applications using Docker and Kubernetes
- Developed machine learning models with TensorFlow

SKILLS
Python, Django, FastAPI, Docker, Kubernetes, Machine Learning, TensorFlow
"""

job_text = """
Senior Python Backend Developer

Requirements:
- 3+ years Python development
- Experience with Django or FastAPI
- Docker and Kubernetes knowledge
- REST API design experience

Skills: Python, Django, FastAPI, Docker, Kubernetes
"""

print("Testing evaluation (this will use skill graph matcher with Groq API)...")
print(f"Resume length: {len(resume_text)} chars")
print(f"Job length: {len(job_text)} chars")
print()

try:
    result = engine.evaluate_texts(resume_text, job_text, use_llm=True)
    print("✓ Evaluation completed successfully!")
    print()
    print(f"Score: {result.score}")
    print(f"Grade: {result.grade}")
    print(f"Semantic Similarity: {result.semantic_similarity}")
    print(f"Keyword Coverage: {result.keyword_coverage}")
    print(f"Section Completeness: {result.section_completeness}")
    print()
    print(f"Keyword Matches: {len(result.keyword_matches)} matches")
    for match in result.keyword_matches[:5]:
        print(f"  - {match.job_keyword} ← {match.resume_keyword} ({match.similarity:.1f}%)")
    print()
    print(f"Missing Keywords: {len(result.missing_keywords)} missing")
    if result.missing_keywords:
        for kw in result.missing_keywords[:3]:
            print(f"  - {kw}")
except Exception as e:
    print(f"✗ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()

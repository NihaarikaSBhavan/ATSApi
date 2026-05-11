"""
Test with improved configuration
Weights tuned for tech-heavy role matching
"""
import logging
from dotenv import load_dotenv
from atsv4 import ATSConfig, ATSEngine

load_dotenv()
logging.basicConfig(level=logging.WARNING)

# Create TUNED config
config = ATSConfig()
config.semantic_weight = 0.40          # Reduce semantic penalty
config.keyword_weight = 0.45            # Boost skills importance  
config.completeness_weight = 0.15       # Reduce section penalty
config.fuzzy_match_threshold = 80.0    # Slightly lower threshold

print("ATSV4 Evaluation - TUNED CONFIGURATION")
print("=" * 70)
print()
print("Configuration:")
print(f"  semantic_weight:        {config.semantic_weight} (was 0.45)")
print(f"  keyword_weight:         {config.keyword_weight} (was 0.35)")
print(f"  completeness_weight:    {config.completeness_weight} (was 0.20)")
print(f"  fuzzy_match_threshold:  {config.fuzzy_match_threshold} (was 84.0)")
print()

# Read test files
with open("test_resume.txt", "r") as f:
    resume_text = f.read()

with open("test_job.txt", "r") as f:
    job_text = f.read()

# Create engine and evaluate
engine = ATSEngine(config)
result = engine.evaluate_texts(resume_text, job_text, use_llm=False)

print("=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)
print()
print(f"Score:                 {result.score} (was 61.56)")
print(f"Grade:                 {result.grade} (was C)")
print(f"Semantic Similarity:   {result.semantic_similarity:.4f} (was 0.4931)")
print(f"Keyword Coverage:      {result.keyword_coverage:.4f} (was 0.8391)")
print(f"Section Completeness:  {result.section_completeness:.4f} (was 0.5000)")
print()

print("=" * 70)
print("SECTION SCORES")
print("=" * 70)
print()
for section, score in result.section_scores.items():
    print(f"  {section:15} {score:.4f}")
print()

print("=" * 70)
print("KEYWORD MATCHING")
print("=" * 70)
print()
print(f"Matches: {len(result.keyword_matches)}")
for match in result.keyword_matches:
    print(f"  - {match.job_keyword:25} <- {match.resume_keyword:25} ({match.similarity:6.1f}%)")
print()

if result.missing_keywords:
    print(f"Missing: {len(result.missing_keywords)}")
    for kw in result.missing_keywords:
        print(f"  - {kw}")
else:
    print("Missing: 0 (Perfect match!)")
print()

print("=" * 70)
print("COMPARISON")
print("=" * 70)
print()
print("               ORIGINAL    TUNED     CHANGE")
print("-" * 48)
print(f"Score:         61.56       {result.score:6.2f}      +{result.score - 61.56:.2f}")
print(f"Keyword Cov:   83.91%      {result.keyword_coverage*100:.2f}%      +{(result.keyword_coverage - 0.8391)*100:.2f}%")

"""
Debug 4: Test the matching algorithm directly
Check why REST APIs isn't being matched despite both having it
"""
from dotenv import load_dotenv
from atsv4 import ATSConfig
from atsv4.skill_graph_matcher import (
    _extract_raw, 
    InMemorySkillGraph, 
    _match_skill
)

load_dotenv()

config = ATSConfig()

resume_text = """John Doe
SKILLS
- Python, Django, FastAPI
- Built REST APIs with proper documentation"""

job_text = """Senior Python Developer
Skills Required:
- Python (expert level)
- Django or FastAPI
- REST APIs
- Docker"""

print("=" * 70)
print("DEBUG 4: SKILL MATCHING ALGORITHM")
print("=" * 70)
print()

# Extract skills
resume_skills = _extract_raw(resume_text, "resume", config)
job_skills = _extract_raw(job_text, "job_description", config)

print(f"Resume skills extracted: {len(resume_skills)}")
for s in resume_skills:
    print(f"  - {s.get('name')}")
print()

print(f"Job skills extracted: {len(job_skills)}")
for s in job_skills:
    print(f"  - {s.get('name')}")
print()

# Build graphs
resume_graph = InMemorySkillGraph()
resume_graph.add_skills(resume_skills)

job_graph = InMemorySkillGraph()
job_graph.add_skills(job_skills)

print(f"Resume graph size: {len(resume_graph.all_skills)} nodes")
print(f"Job graph size: {len(job_graph.all_skills)} nodes")
print()

# Match each job skill
threshold = getattr(config, "fuzzy_match_threshold", 84.0)
print(f"Fuzzy match threshold: {threshold}")
print()

print("Matching Results:")
print("-" * 70)
for job_skill_node in job_graph.all_skills:
    result = _match_skill(job_skill_node, resume_graph, threshold)
    status = "MATCH" if result.match_type != "none" else "MISS"
    print(f"{status}: Job '{result.job_skill}'")
    print(f"   Type: {result.match_type}")
    print(f"   Resume: '{result.resume_skill}'")
    print(f"   Sim: {result.similarity:.1f}%")
    if result.relation_path:
        print(f"   Path: {result.relation_path}")
    print()

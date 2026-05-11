"""
Debug 6: Test with ACTUAL test files
Use the real test_resume.txt and test_job.txt to see where REST APIs issue happens
"""
from dotenv import load_dotenv
from atsv4 import ATSConfig
from atsv4.skill_graph_matcher import _extract_raw, InMemorySkillGraph, _match_skill

load_dotenv()

config = ATSConfig()

# Read actual test files
with open("test_resume.txt", "r") as f:
    resume_text = f.read()

with open("test_job.txt", "r") as f:
    job_text = f.read()

print("=" * 70)
print("DEBUG 6: ACTUAL FILE TEST")
print("=" * 70)
print()

# Extract skills
print("Extracting from ACTUAL RESUME...")
resume_skills = _extract_raw(resume_text, "resume", config)
print(f"Total: {len(resume_skills)} skills")
resume_skill_names = [s.get('name') for s in resume_skills]
for name in sorted(resume_skill_names):
    print(f"  - {name}")
print()

print("Extracting from ACTUAL JOB...")
job_skills = _extract_raw(job_text, "job_description", config)
print(f"Total: {len(job_skills)} skills")
job_skill_names = [s.get('name') for s in job_skills]
for name in sorted(job_skill_names):
    print(f"  - {name}")
print()

# Build graphs
resume_graph = InMemorySkillGraph()
resume_graph.add_skills(resume_skills)

job_graph = InMemorySkillGraph()
job_graph.add_skills(job_skills)

print("=" * 70)
print("MATCHING EACH JOB SKILL")
print("=" * 70)
print()

threshold = config.fuzzy_match_threshold
matches = []
missing = []

for job_skill_node in job_graph.all_skills:
    result = _match_skill(job_skill_node, resume_graph, threshold)
    if result.match_type == "none":
        missing.append(result.job_skill)
    else:
        matches.append((result.job_skill, result.resume_skill, result.similarity, result.match_type))

print(f"MATCHED ({len(matches)}):")
for job_kw, resume_kw, sim, match_type in sorted(matches):
    print(f"  - {job_kw:25} <- {resume_kw:25} ({sim:6.1f}%, {match_type})")
print()

print(f"MISSING ({len(missing)}):")
for kw in sorted(missing):
    print(f"  - {kw}")
print()

# Find REST
print("=" * 70)
print("REST APIs ANALYSIS")
print("=" * 70)
print()

rest_in_resume = [s for s in resume_skills if 'REST' in s.get('name', '').upper()]
rest_in_job = [s for s in job_skills if 'REST' in s.get('name', '').upper()]

print(f"Found 'REST' in resume skills: {len(rest_in_resume)}")
if rest_in_resume:
    for s in rest_in_resume:
        print(f"  - Exact name: '{s.get('name')}'")
        print(f"    Level: {s.get('level')}, Category: {s.get('category')}")
else:
    print("  [NOT FOUND]")
print()

print(f"Found 'REST' in job skills: {len(rest_in_job)}")
if rest_in_job:
    for s in rest_in_job:
        print(f"  - Exact name: '{s.get('name')}'")
        print(f"    Level: {s.get('level')}, Category: {s.get('category')}")
else:
    print("  [NOT FOUND]")
print()

# Check if they match
if rest_in_resume and rest_in_job:
    job_rest_node = job_graph.get(rest_in_job[0].get('name'))
    if job_rest_node:
        result = _match_skill(job_rest_node, resume_graph, threshold)
        print(f"Attempting to match job '{rest_in_job[0].get('name')}':")
        print(f"  Match Type: {result.match_type}")
        print(f"  Resume Skill: {result.resume_skill}")
        print(f"  Similarity: {result.similarity:.1f}%")

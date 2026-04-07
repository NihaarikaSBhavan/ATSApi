"""
ATS Dataset Validation + Processing Pipeline
-------------------------------------------
Input: Raw JSON dataset (from Claude or any source)
Output: Clean, validated dataset ready for training
"""

import json
import pandas as pd
import numpy as np

# ==============================
# CONFIG
# ==============================

INPUT_FILE = "C:\\Users\\Admin\\Desktop\\ATS_sys\\ATSProd\\Data\\raw_dataset.json"   # your Claude output
OUTPUT_FILE = "clean_ats_dataset.csv"

VALID_DOMAINS = {
    "tech","healthcare","finance","manufacturing",
    "sales_marketing","logistics","hr",
    "education","construction","legal","generic"
}


# ==============================
# LOAD DATA
# ==============================

def load_dataset(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ==============================
# CLEANING
# ==============================

def clean_dataset(data):
    cleaned = []

    for sample in data:
        try:
            cleaned.append({
                "resume_text": str(sample["resume_text"]),
                "job_description": str(sample["job_description"]),
                "domain": str(sample["domain"]).lower(),
                "matched_skills": sample.get("matched_skills", []),
                "missing_skills": sample.get("missing_skills", []),
                "inferred_skills": sample.get("inferred_skills", []),
                "semantic_similarity_score": float(sample["semantic_similarity_score"]),
                "keyword_overlap_score": float(sample["keyword_overlap_score"]),
                "final_score": float(sample["final_score"]),
                "match_label": str(sample["match_label"])
            })
        except:
            continue

    return cleaned


# ==============================
# VALIDATION FUNCTIONS
# ==============================

def validate_domain(sample):
    return sample["domain"] in VALID_DOMAINS


def validate_score(sample):
    score = sample["final_score"]

    if score < 30:
        return sample["match_label"] == "Poor"
    elif score < 70:
        return sample["match_label"] == "Moderate"
    else:
        return sample["match_label"] == "Strong"


def validate_skills(sample):
    return (
        isinstance(sample["matched_skills"], list) and
        isinstance(sample["missing_skills"], list) and
        isinstance(sample["inferred_skills"], list)
    )


def validate_text(sample):
    return (
        len(sample["resume_text"]) > 50 and
        len(sample["job_description"]) > 50
    )


# ==============================
# FEATURE RECOMPUTATION
# ==============================

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")


def compute_global_similarity(resume_text, jd_text):
    r = model.encode([resume_text], normalize_embeddings=True)
    j = model.encode([jd_text], normalize_embeddings=True)
    return cosine_similarity(r, j)[0][0]


def keyword_overlap(resume_text, jd_text):
    r_words = set(resume_text.lower().split())
    j_words = set(jd_text.lower().split())

    overlap = len(r_words & j_words) / max(len(j_words), 1)
    return overlap


def validate_consistency(sample):
    emb = compute_global_similarity(
        sample["resume_text"],
        sample["job_description"]
    )

    kw = keyword_overlap(
        sample["resume_text"],
        sample["job_description"]
    )

    # Consistency rules
    if sample["semantic_similarity_score"] > 0.7 and emb < 0.3:
        return False

    if sample["keyword_overlap_score"] > 0.5 and kw < 0.1:
        return False

    return True


# ==============================
# FULL VALIDATION PIPELINE
# ==============================

def validate_dataset(dataset):
    valid = []
    rejected = []

    for sample in dataset:
        if not validate_domain(sample):
            rejected.append(("domain", sample))
            continue

        if not validate_score(sample):
            rejected.append(("score", sample))
            continue

        if not validate_skills(sample):
            rejected.append(("skills", sample))
            continue

        if not validate_text(sample):
            rejected.append(("text", sample))
            continue

        if not validate_consistency(sample):
            rejected.append(("consistency", sample))
            continue

        valid.append(sample)

    return valid, rejected


# ==============================
# SAVE DATASET
# ==============================

def save_dataset(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"✅ Saved {len(data)} samples to {filename}")


# ==============================
# MAIN
# ==============================

def run_pipeline():
    raw_data = load_dataset(INPUT_FILE)
    print(f"Loaded {len(raw_data)} raw samples")

    cleaned = clean_dataset(raw_data)
    print(f"After cleaning: {len(cleaned)} samples")

    valid, rejected = validate_dataset(cleaned)
    print(f"Valid samples: {len(valid)}")
    print(f"Rejected samples: {len(rejected)}")

    save_dataset(valid, OUTPUT_FILE)

    return valid, rejected


if __name__ == "__main__":
    run_pipeline()
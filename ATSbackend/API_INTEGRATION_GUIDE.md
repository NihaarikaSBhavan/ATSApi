## API Integration Guide - Semantic ATS Core

**Objective**: Integrate the new semantic model into the REST API for production use.

---

## 1. Load Trained Model in Predictor

**File**: `backend/scoring/predictor.py`

```python
# Add at module level (after existing imports)
import xgboost as xgb
import numpy as np
from pathlib import Path

# Load trained semantic model (call once at startup)
_SEMANTIC_MODEL = None
_FEATURE_SCALER = None

def load_semantic_model():
    """Load trained XGBoost model and scaler."""
    global _SEMANTIC_MODEL, _FEATURE_SCALER
    
    model_path = Path(__file__).parent.parent.parent / 'models' / 'semantic_xgboost_model.bin'
    scaler_path = Path(__file__).parent.parent.parent / 'models' / 'feature_scaler.npy'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    _SEMANTIC_MODEL = xgb.Booster()
    _SEMANTIC_MODEL.load_model(str(model_path))
    
    # Load scaler (saved as [mean_, scale_])
    scaler_data = np.load(str(scaler_path), allow_pickle=True)
    _FEATURE_SCALER = {
        'mean': scaler_data[0],
        'scale': scaler_data[1]
    }
    
    return _SEMANTIC_MODEL
```

---

## 2. Update Predict Score Function

```python
def predict_score(features):
    """
    Predict match score using trained semantic XGBoost model.
    
    Args:
        features: 18-feature numpy array from build_features()
    
    Returns:
        score: [0, 100] match score
    """
    global _SEMANTIC_MODEL, _FEATURE_SCALER
    
    if _SEMANTIC_MODEL is None:
        load_semantic_model()
    
    # Standardize features using saved scaler
    features_scaled = (features - _FEATURE_SCALER['mean']) / _FEATURE_SCALER['scale']
    
    # Create DMatrix
    dmatrix = xgb.DMatrix(features_scaled.reshape(1, -1))
    
    # Predict (returns class probabilities for multi:softmax)
    # For continuous score, we'll use class prediction + confidence
    prediction = _SEMANTIC_MODEL.predict(dmatrix)[0]  # 0=Poor, 1=Moderate, 2=Strong
    
    # Convert class to score
    class_to_score = {
        0: 25,  # Poor
        1: 60,  # Moderate
        2: 85   # Strong
    }
    
    return float(class_to_score.get(int(prediction), 50))
```

---

## 3. Update API Response with Semantic Details

**File**: `backend/api/routes.py`

```python
from backend.features.feature_engineer import build_features

@router.post("/evaluate")
async def evaluate_resume_jd(
    resume_text: str = Form(...),
    jd_text: str = Form(...)
):
    """
    Evaluate resume against job description.
    Now includes semantic feature breakdown.
    """
    try:
        # Extract skills
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(jd_text)
        
        # Domain detection
        detected_domain = detect_domain(jd_text)
        
        # Build graph
        graph = SkillGraph(domain=detected_domain)
        graph.build([resume_skills, jd_skills])
        
        # Extract all 18 features (NEW)
        features = build_features(resume_text, jd_text, resume_skills, jd_skills, graph)
        
        # Predict score
        score = predict_score(features)
        
        # Build response with semantic details
        return {
            "score": round(score, 1),
            "label": score_label(score),
            "domain": detected_domain,
            
            # NEW: Semantic feature breakdown
            "semantic_features": {
                "capability_score": round(float(features[0]), 3),
                "embedding_similarity": round(float(features[1]), 3),
                "keyword_overlap": round(float(features[2]), 3),
                "graph_match": round(float(features[3]), 3),
                "inferred_coverage": round(float(features[4]), 3),
            },
            
            # Matched/missing skills
            "matched_skills": sorted(set(resume_skills) & set(jd_skills)),
            "missing_skills": sorted(set(jd_skills) - set(resume_skills)),
            "inferred_skills": graph.infer_skills(resume_skills, top_k=5),
            
            # Explainability
            "explanation": _generate_explanation(
                score, features, resume_skills, jd_skills
            ),
        }
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise

def _generate_explanation(score, features, resume_skills, jd_skills):
    """Generate human-readable explanation."""
    f0_capability = features[0]
    f1_embedding = features[1]
    f2_keyword = features[2]
    
    explanation = []
    
    if f0_capability > 0.7:
        explanation.append("✓ Strong semantic skill match (embeddings align well)")
    elif f0_capability > 0.4:
        explanation.append("~ Moderate semantic skill match (some gaps)")
    else:
        explanation.append("⚠ Weak semantic skill match (major gaps)")
    
    if f1_embedding > 0.6:
        explanation.append("✓ Resume and JD are semantically related")
    
    if f2_keyword > 0.5:
        explanation.append(f"✓ {int(len(set(resume_skills) & set(jd_skills)))} exact keyword matches")
    else:
        explanation.append("~ Few exact keyword matches (but semantic similarity detected)")
    
    if score >= 70:
        explanation.append("→ Candidate is a strong fit")
    elif score >= 50:
        explanation.append("→ Candidate has relevant skills; review for gaps")
    else:
        explanation.append("→ Candidate may lack key requirements")
    
    return " ".join(explanation)
```

---

## 4. Startup Sequence (main.py)

```python
# Add to main.py startup
import logging
from backend.scoring.predictor import load_semantic_model

logger = logging.getLogger(__name__)

@asyncio.event_handler()
async def startup_event():
    """Load models at startup."""
    logger.info("Loading semantic XGBoost model...")
    try:
        load_semantic_model()
        logger.info("✓ Semantic model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("Falling back to feature-based scoring")
```

---

## 5. Testing Integration

```bash
# Test API locally
curl -X POST http://localhost:8000/evaluate \
  -F "resume_text=Experienced Python developer with ML expertise" \
  -F "jd_text=Seeking senior Python developer for ML/AI projects"

# Expected response:
{
  "score": 85.0,
  "label": "Strong",
  "domain": "tech",
  "semantic_features": {
    "capability_score": 0.82,
    "embedding_similarity": 0.78,
    "keyword_overlap": 1.0,
    "graph_match": 0.71,
    "inferred_coverage": 0.5
  },
  "matched_skills": ["python", "machine learning"],
  "missing_skills": [],
  "inferred_skills": ["deep learning", "tensorflow", "pytorch"],
  "explanation": "✓ Strong semantic skill match. ✓ Resume and JD are semantically related. ✓ 2 exact keyword matches → Candidate is a strong fit"
}
```

---

## 6. Fallback Strategy (if model unavailable)

```python
def predict_score_fallback(features):
    """
    Fallback scoring if model not loaded.
    Uses feature-based heuristic.
    """
    f0_capability = features[0]  # 0.35 weight
    f1_embedding = features[1]   # 0.25 weight
    f3_graph = features[3]       # 0.20 weight
    f4_inferred = features[4]    # 0.10 weight
    f2_keyword = features[2]     # 0.10 weight
    
    # Weighted combination
    score = (
        0.35 * f0_capability +
        0.25 * f1_embedding +
        0.20 * f3_graph +
        0.10 * f4_inferred +
        0.10 * f2_keyword
    )
    
    # Scale to [0, 100]
    return max(0, min(100, score * 100))
```

---

## 7. Monitoring & Metrics

Add to monitoring dashboard:

```python
# Track feature usage in production
metrics = {
    "capability_score_avg": np.mean([f[0] for f in features_cache]),
    "embedding_similarity_avg": np.mean([f[1] for f in features_cache]),
    "model_confidence": np.mean([max_pred_prob for _ in predictions]),
    "score_distribution": {
        "poor": percent_of_poor,
        "moderate": percent_of_moderate,
        "strong": percent_of_strong,
    }
}
```

---

## 8. Rollout Plan

### Phase 1: Shadow Mode (1 day)
- Load model but dont change predictions
- Log new scores alongside old
- Compare performance

### Phase 2: Canary Deployment (2-3 days)
- Route 10% of traffic to new model
- Monitor error rates, latency
- Gather user feedback

### Phase 3: Full Rollout (1 day)
- Switch 100% traffic to semantic model
- Keep old model as fallback
- Monitor metrics

### Phase 4: Optimize (ongoing)
- Collect feedback on score quality
- Fine-tune thresholds
- Plan domain-specific models

---

## 9. Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Check `models/semantic_xgboost_model.bin` exists |
| Slow predictions | Add feature caching for common resume/JD pairs |
| Low accuracy on domain X | Collect more domain X samples, retrain |
| Scores too high/low | Recalibrate thresholds or retrain with better labels |
| Memory issues | Use batch prediction, clean old cached embeddings |

---

## 10. Rollback Procedure

```bash
# If issues arise, revert to old scoring
# 1. Remove load_semantic_model() call in main.py
# 2. Restore old predict_score() function
# 3. Restart API
# 4. Monitor metrics

# Keep model file for later investigation
cp models/semantic_xgboost_model.bin models/semantic_xgboost_model.bin.backup
```

---

**Next**: Proceed to Phase 2B (spaCy NER integration) after confirming API integration works.

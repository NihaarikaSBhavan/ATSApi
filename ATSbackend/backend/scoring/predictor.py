"""
ATS Scorer — Liberal Probabilistic Continuous Scoring

Replaces the rigid 3-class XGBoost (outputs only 25/60/85) with a smooth
sigmoid over a weighted composite of the 18-element feature vector.

Design goals
------------
* Liberal: a candidate who partially meets requirements scores 50-65, not 25.
* Probabilistic: every marginal improvement shifts the score proportionally.
* Realistic: perfect matches score 88-95; zero-overlap scores 8-15.
* No model file required — formula is deterministic and version-controlled.
"""

import math
import logging
import numpy as np
from typing import Union

logger = logging.getLogger(__name__)

_CENTRE    = 0.42
_STEEPNESS = 7.0

_W = {
    0:  0.28,
    1:  0.18,
    2:  0.12,
    3:  0.18,
    4:  0.08,
    6:  0.07,
    7:  0.05,
    10: 0.04,
}


def _safe(features, i, default=0.0):
    return float(features[i]) if len(features) > i else default


def _composite(features):
    raw = sum(_W[i] * _safe(features, i) for i in _W)
    f1  = _safe(features, 1)
    f2  = _safe(features, 2)
    f17 = _safe(features, 17, 1.0)
    if f17 > 0.5 and f1 > 0.35 and f2 < 0.10:
        raw = min(raw + 0.06, 1.0)
    f12 = _safe(features, 12)
    f13 = _safe(features, 13)
    if f12 > 0.5 or f13 > 0.5:
        raw *= 0.4
    return float(np.clip(raw, 0.0, 1.0))


def _sigmoid(x):
    try:
        return 1.0 / (1.0 + math.exp(-_STEEPNESS * (x - _CENTRE)))
    except OverflowError:
        return 0.0 if x < _CENTRE else 1.0


def predict_score(features):
    if not isinstance(features, np.ndarray):
        features = np.array(features, dtype=float)
    try:
        comp  = _composite(features)
        score = float(np.clip(_sigmoid(comp) * 100.0, 0.0, 100.0))
        logger.info(
            f"[predictor] composite={comp:.4f} score={score:.1f} "
            f"f0={_safe(features,0):.3f} f1={_safe(features,1):.3f} "
            f"f2={_safe(features,2):.3f} f3={_safe(features,3):.3f}"
        )
        return round(score, 1)
    except Exception as exc:
        logger.error(f"[predictor] Scoring failed: {exc}.", exc_info=True)
        return 50.0


def score_label(score):
    if score >= 80: return "Excellent Match"
    if score >= 67: return "Strong Match"
    if score >= 52: return "Good Match"
    if score >= 38: return "Moderate Match"
    if score >= 22: return "Weak Match"
    return "Poor Match"

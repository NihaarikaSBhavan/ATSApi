import os
import threading
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
import logging

logger = logging.getLogger(__name__)

# Use joblib for safer serialization
try:
    from joblib import load as joblib_load, dump as joblib_dump
    USE_JOBLIB = True
except ImportError:
    import pickle
    USE_JOBLIB = False

# Import XGBoost if available (preferred model)
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed. Install with: pip install xgboost")

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ats_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'ats_scaler.pkl')

# Thread-safe model caching
_model_cache = None
_scaler_cache = None
_model_lock = threading.RLock()

# 17 features: f0-f5=semantic, f6-f11=contextual, f12-f15=binary missingness, f16=domain_flag
# f0=global_sim, f1=skill_sim, f2=exact_overlap, f3=graph_soft, f4=inferred_coverage,
# f5=composite, f6=exp_level, f7=education, f8=certifications, f9=ecosystem,
# f10=keyword_density, f11=recency,
# f12=is_resume_empty, f13=is_skills_empty, f14=has_exact_overlap, f15=has_inferred, f16=domain_flag
SEED = np.array([
    # [f0,   f1,   f2,   f3,   f4,   f5,   f6,  f7,  f8,  f9,  f10, f11, f12, f13, f14, f15, score]
    # Perfect matches (high on everything, no missingness)
    [0.95, 0.93, 0.90, 0.92, 0.90, 0.929, 0.95, 0.90, 0.85, 0.88, 0.85, 0.92, 0, 0, 1, 1, 0, 96],
    [0.90, 0.88, 0.85, 0.88, 0.85, 0.884, 0.85, 0.85, 0.75, 0.82, 0.80, 0.88, 0, 0, 1, 1, 0, 91],
    [0.85, 0.84, 0.80, 0.85, 0.80, 0.842, 0.80, 0.80, 0.70, 0.78, 0.75, 0.85, 0, 0, 1, 1, 0, 87],
    # Strong semantic match, partial exact (has semantic, some missing exact)
    [0.75, 0.80, 0.35, 0.78, 0.65, 0.736, 0.72, 0.70, 0.60, 0.72, 0.70, 0.78, 0, 0, 1, 1, 0, 78],
    [0.70, 0.76, 0.28, 0.74, 0.60, 0.700, 0.68, 0.65, 0.55, 0.68, 0.65, 0.75, 0, 0, 1, 1, 0, 74],
    [0.68, 0.72, 0.25, 0.70, 0.55, 0.672, 0.65, 0.60, 0.50, 0.65, 0.60, 0.72, 0, 0, 1, 1, 0, 71],
    # Moderate (good semantic, limited exact, has inference)
    [0.65, 0.65, 0.35, 0.62, 0.45, 0.624, 0.58, 0.55, 0.45, 0.60, 0.58, 0.68, 0, 0, 1, 1, 0, 66],
    [0.60, 0.60, 0.30, 0.58, 0.40, 0.578, 0.55, 0.50, 0.40, 0.55, 0.55, 0.65, 0, 0, 1, 1, 0, 61],
    [0.58, 0.55, 0.27, 0.55, 0.35, 0.547, 0.52, 0.48, 0.38, 0.52, 0.52, 0.62, 0, 0, 1, 1, 0, 57],
    # Weak matches (low semantic, minimal exact)
    [0.50, 0.45, 0.20, 0.45, 0.25, 0.452, 0.45, 0.40, 0.30, 0.45, 0.45, 0.55, 0, 0, 1, 0, 0, 47],
    [0.40, 0.35, 0.15, 0.35, 0.20, 0.355, 0.38, 0.35, 0.25, 0.38, 0.38, 0.48, 0, 0, 1, 0, 0, 37],
    [0.30, 0.25, 0.10, 0.25, 0.10, 0.255, 0.32, 0.30, 0.20, 0.32, 0.32, 0.42, 0, 0, 0, 0, 0, 26],
    # Poor matches (very low all)
    [0.20, 0.15, 0.05, 0.15, 0.05, 0.155, 0.25, 0.22, 0.15, 0.25, 0.25, 0.35, 0, 0, 0, 0, 0, 16],
    [0.10, 0.08, 0.02, 0.08, 0.02, 0.078, 0.15, 0.12, 0.08, 0.15, 0.15, 0.25, 0, 0, 0, 0, 0, 9],
    # Edge case: high exact but low semantic (keyword-heavy, low text similarity)
    [0.45, 0.40, 0.70, 0.65, 0.50, 0.500, 0.40, 0.38, 0.42, 0.38, 0.40, 0.45, 0, 0, 1, 1, 0, 58],
    # Edge case: high semantic, zero exact (transferable skills, low keyword overlap)
    [0.72, 0.78, 0.05, 0.72, 0.60, 0.680, 0.70, 0.68, 0.52, 0.70, 0.68, 0.80, 0, 0, 0, 1, 0, 72],
    # Sparse data case: zero exact, zero inferred (weak signal from all channels)
    [0.35, 0.32, 0.00, 0.30, 0.00, 0.320, 0.40, 0.35, 0.30, 0.35, 0.30, 0.45, 0, 0, 0, 0, 0, 20],
])

def create_model():
    """
    Create best available model (XGBoost > GradientBoosting).
    
    XGBoost config:
    - max_depth=4: Shallow trees to reduce overfitting
    - reg_alpha=1.0: L1 regularization (feature selection)
    - reg_lambda=2.0: L2 regularization (weight smoothing)
    - colsample_bytree=0.8: Feature subsampling per tree
    - subsample=0.8: Row subsampling (stochastic boosting)
    - learning_rate=0.05: Conservative step size
    - n_estimators=500: Sufficient trees with early stopping
    """
    if HAS_XGBOOST:
        logger.info("Using XGBoost model with L1/L2 regularization")
        return XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            reg_alpha=1.0,        # L1 regularization
            reg_lambda=2.0,       # L2 regularization
            subsample=0.8,        # Row subsampling
            colsample_bytree=0.8, # Feature subsampling
            random_state=42,
            verbosity=0
        )
    else:
        logger.info("Using GradientBoosting model")
        return GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.8,
            random_state=42
        )

def train_model(extra_X=None, extra_y=None):
    """
    Train and cache the model with cross-validation.
    
    Now uses 17-feature vector:
    - Features 0-11: Semantic, contextual, compound features
    - Features 12-15: Binary missingness indicators
    - Feature 16: Domain flag (generic vs domain-specific)
    """
    global _model_cache, _scaler_cache
    
    X, y = SEED[:, :17], SEED[:, 17]  # Use 17 features + score column
    if extra_X is not None:
        X, y = np.vstack([X, extra_X]), np.concatenate([y, extra_y])
    
    sc = MinMaxScaler()
    X_scaled = sc.fit_transform(X)
    
    m = create_model()
    m.fit(X_scaled, y)
    
    # Log cross-validation score
    cv_scores = cross_val_score(m, X_scaled, y, cv=5, scoring='r2')
    logger.info(f"Model CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Save to disk
    if USE_JOBLIB:
        joblib_dump(m, MODEL_PATH)
        joblib_dump(sc, SCALER_PATH)
    else:
        with open(MODEL_PATH, 'wb') as f:
            import pickle
            pickle.dump(m, f)
        with open(SCALER_PATH, 'wb') as f:
            import pickle
            pickle.dump(sc, f)
    
    # Update cache
    with _model_lock:
        _model_cache = m
        _scaler_cache = sc
    
    logger.info("Model trained and cached successfully.")
    return m


def load_model():
    """Load model and scaler with thread-safe caching."""
    global _model_cache, _scaler_cache
    
    # Return cached models if available
    with _model_lock:
        if _model_cache is not None and _scaler_cache is not None:
            return _model_cache, _scaler_cache
    
    # Train if not found
    if not os.path.exists(MODEL_PATH):
        return train_model()
    
    # Load from disk with thread safety
    with _model_lock:
        if USE_JOBLIB:
            m = joblib_load(MODEL_PATH)
            sc = joblib_load(SCALER_PATH)
        else:
            import pickle
            with open(MODEL_PATH, 'rb') as f:
                m = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                sc = pickle.load(f)
        
        # Validate scaler feature count matches current feature vector size (16)
        try:
            if getattr(sc, 'n_features_in_', None) is not None and sc.n_features_in_ != 17:
                logger.warning("Scaler expects %d features but current pipeline uses 17; retraining model.", sc.n_features_in_)
                return train_model()
        except Exception:
            logger.debug("Could not validate scaler feature count; proceeding to cache.")

        # Cache for future calls
        _model_cache = m
        _scaler_cache = sc
    
    return m, sc

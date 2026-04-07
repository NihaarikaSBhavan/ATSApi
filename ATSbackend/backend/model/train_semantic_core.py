"""
Semantic ATS Core Training and Validation
---
Trains XGBoost on the semantic features including the new capability_score.
Uses cross-domain labeled data from clean_ats_dataset.csv.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from backend.extraction.skill_extractor import extract_skills
from backend.graph.skill_graph import detect_domain, SkillGraph
from backend.features.feature_engineer import build_features
from backend.scoring.predictor import predict_score, score_label

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Label mapping
LABEL_MAPPING = {
    'Poor': 0,
    'Moderate': 1,
    'Strong': 2
}

INVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}


def load_dataset(csv_path):
    """Load clean_ats_dataset.csv and prepare for training."""
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} samples from {csv_path}")
    logger.info(f"Domains: {df['domain'].unique()}")
    logger.info(f"Labels: {df['match_label'].value_counts().to_dict()}")
    return df


def extract_features_from_row(row):
    """Extract all features for a single resume/JD pair."""
    resume_text = row['resume_text']
    jd_text = row['job_description']
    domain = row['domain']
    
    try:
        # Extract skills
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(jd_text)
        
        # Build graph
        detected = detect_domain(jd_text)
        # Use ground-truth domain if available, else detected
        graph = SkillGraph(domain=domain if domain != 'generic' else detected)
        graph.build([resume_skills, jd_skills])
        
        # Extract features
        features = build_features(resume_text, jd_text, resume_skills, jd_skills, graph)
        
        return features, None
    except Exception as e:
        logger.error(f"Error processing row: {e}")
        return None, e


def prepare_training_data(df):
    """Prepare feature matrix and labels."""
    X = []
    y = []
    failed = 0
    
    for idx, row in df.iterrows():
        features, error = extract_features_from_row(row)
        if error:
            failed += 1
            continue
        
        X.append(features)
        label = LABEL_MAPPING.get(row['match_label'], 1)  # Default to Moderate
        y.append(label)
    
    logger.info(f"Prepared {len(X)} samples ({failed} failed)")
    return np.array(X), np.array(y)


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model on semantic features."""
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train XGBoost
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dval = xgb.DMatrix(X_val_scaled, label=y_val)
    
    params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'max_depth': 5,
        'learning_rate': 0.1,
        'n_estimators': 100,
    }
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dval, 'val')],
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    return model, scaler


def evaluate_model(model, scaler, X_test, y_test):
    """Evaluate model performance."""
    X_test_scaled = scaler.transform(X_test)
    dtest = xgb.DMatrix(X_test_scaled)
    
    y_pred = model.predict(dtest).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    logger.info(f"\nTest Set Performance:")
    logger.info(f"  Accuracy:  {acc:.3f}")
    logger.info(f"  Precision: {prec:.3f}")
    logger.info(f"  Recall:    {rec:.3f}")
    logger.info(f"  F1 Score:  {f1:.3f}")
    
    # Per-class breakdown
    for label_idx, label_name in INVERSE_LABEL_MAPPING.items():
        mask = y_test == label_idx
        if mask.sum() > 0:
            class_acc = accuracy_score(y_test[mask], y_pred[mask])
            logger.info(f"  {label_name}: {class_acc:.3f} ({mask.sum()} samples)")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def print_feature_importance(model):
    """Print feature importance."""
    importance = model.get_score(importance_type='weight')
    feature_names = [
        'capability_score',
        'embedding_similarity',
        'keyword_overlap',
        'graph_match',
        'inferred_coverage',
        'composite',
        'experience_alignment',
        'education_match',
        'certifications',
        'ecosystem_coherence',
        'keyword_density',
        'skill_recency',
        'resume_empty',
        'skills_empty',
        'has_overlap',
        'has_inferred',
        'domain_flag',
        'anti_collapse_guardrail',
    ]
    
    logger.info("\n📊 Feature Importance (Top 10):")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feat_idx, score) in enumerate(sorted_imp[:10], 1):
        feat_name = feature_names[int(feat_idx[1:])] if feat_idx.startswith('f') else feat_idx
        logger.info(f"  {i:2d}. {feat_name:<30s} {int(score):6d}")


def main():
    """Train and validate semantic ATS core."""
    
    # Paths
    data_dir = Path(__file__).parent.parent.parent / 'Data'
    csv_path = data_dir / 'clean_ats_dataset.csv'
    
    if not csv_path.exists():
        logger.error(f"Dataset not found: {csv_path}")
        return
    
    # Load data
    df = load_dataset(csv_path)
    
    # Prepare features
    logger.info("\n🔄 Extracting features...")
    X, y = prepare_training_data(df)
    
    if len(X) < 10:
        logger.error("Not enough data to train")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train model
    logger.info("\n🤖 Training XGBoost...")
    model, scaler = train_xgboost(X_train, y_train, X_val, y_val)
    
    # Evaluate
    logger.info("\n📈 Evaluating...")
    evaluate_model(model, scaler, X_test, y_test)
    
    # Feature importance
    print_feature_importance(model)
    
    # Save model
    logger.info("\n💾 Saving model...")
    model_path = Path(__file__).parent.parent.parent / 'models'
    model_path.mkdir(exist_ok=True)
    model.save_model(str(model_path / 'semantic_xgboost_model.bin'))
    np.save(str(model_path / 'feature_scaler.npy'), [scaler.mean_, scaler.scale_])
    
    logger.info(f"✅ Model saved to {model_path}")


if __name__ == '__main__':
    main()

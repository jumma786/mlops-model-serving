"""
Model Trainer for FastAPI Serving
===================================
Trains the tuned XGBoost champion from Project 4 on real UCI Bank Marketing data.
Saves model + feature order + input schema as artifacts for the API.

Usage:
    python src/model/train.py
    python src/model/train.py --data-path data/bank-additional-full.csv
"""

import os
import sys
import json
import logging
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "artifacts"
BEST_PARAMS = {
    "n_estimators":     349,
    "max_depth":        5,
    "learning_rate":    0.0124,
    "subsample":        0.7244,
    "colsample_bytree": 0.7301,
    "min_child_weight": 8,
    "reg_alpha":        0.00126,
    "reg_lambda":       0.1252,
    "scale_pos_weight": 10,
    "eval_metric":      "logloss",
    "verbosity":        0,
    "random_state":     42,
}


def load_and_preprocess(data_path: str = None, n_samples: int = 5000, random_state: int = 42):
    """Load and preprocess UCI Bank Marketing data."""
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading real data: {data_path}")
        df = pd.read_csv(data_path, sep=";")
        df["y"] = (df["y"] == "yes").astype(int)
    else:
        logger.info("Generating synthetic data (real data not found)")
        np.random.seed(random_state)
        n = n_samples
        df = pd.DataFrame({
            "age":           np.random.randint(18, 95, n),
            "job":           np.random.choice(["admin.","blue-collar","management","retired","technician","unknown"], n),
            "marital":       np.random.choice(["divorced","married","single"], n),
            "education":     np.random.choice(["basic.4y","high.school","university.degree","unknown"], n),
            "default":       np.random.choice(["no","yes","unknown"], n, p=[0.79,0.01,0.20]),
            "housing":       np.random.choice(["no","yes"], n),
            "loan":          np.random.choice(["no","yes"], n, p=[0.82,0.18]),
            "contact":       np.random.choice(["cellular","telephone"], n, p=[0.63,0.37]),
            "month":         np.random.choice(["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"], n),
            "day_of_week":   np.random.choice(["mon","tue","wed","thu","fri"], n),
            "campaign":      np.random.randint(1, 15, n),
            "pdays":         np.where(np.random.rand(n) < 0.13, np.random.randint(1,30,n), 999),
            "previous":      np.random.randint(0, 7, n),
            "poutcome":      np.random.choice(["failure","nonexistent","success"], n, p=[0.10,0.86,0.04]),
            "emp.var.rate":  np.random.choice([-1.8,-1.7,1.1,1.4], n),
            "cons.price.idx": np.random.uniform(92.2, 94.8, n).round(3),
            "cons.conf.idx": np.random.uniform(-50.8, -26.9, n).round(1),
            "euribor3m":     np.random.uniform(0.6, 5.1, n).round(3),
            "nr.employed":   np.random.choice([4963.6,5008.7,5099.1,5176.3,5228.1], n),
            "y":             (np.random.rand(n) < 0.11).astype(int),
        })

    # Drop leakage feature
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])
        logger.info("Dropped 'duration' — leakage feature")

    # Encode categoricals
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = {v: int(i) for i, v in enumerate(le.classes_)}

    X = df.drop(columns=["y"])
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape} | Positive: {y_train.mean():.1%}")
    return X_train, X_test, y_train, y_test, list(X.columns), encoders


def train_and_save(data_path: str = None, random_state: int = 42):
    """Train champion model and save all artifacts."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test, feature_names, encoders = load_and_preprocess(
        data_path=data_path, random_state=random_state
    )

    # Train champion model
    logger.info("Training XGBoost champion (Project 4 best params)...")
    model = xgb.XGBClassifier(**BEST_PARAMS)
    model.fit(X_train, y_train)

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_prob)
    f1  = f1_score(y_test, y_pred, zero_division=0)

    logger.info(f"Test AUC: {auc:.4f} | F1: {f1:.4f}")

    # Save model
    model_path = os.path.join(ARTIFACTS_DIR, "champion_model.json")
    model.save_model(model_path)
    logger.info(f"Model saved: {model_path}")

    # Save feature order
    feature_path = os.path.join(ARTIFACTS_DIR, "feature_order.json")
    with open(feature_path, "w") as f:
        json.dump(feature_names, f)
    logger.info(f"Feature order saved: {feature_path}")

    # Save encoders
    encoder_path = os.path.join(ARTIFACTS_DIR, "encoders.json")
    with open(encoder_path, "w") as f:
        json.dump(encoders, f)
    logger.info(f"Encoders saved: {encoder_path}")

    # Save model info
    info = {
        "model_type":   "XGBoost",
        "auc":          round(auc, 4),
        "f1":           round(f1, 4),
        "n_features":   len(feature_names),
        "features":     feature_names,
        "baseline_auc": 0.8174,
        "project":      "mlops-model-serving (Project 5)",
        "data_source":  data_path if data_path else "synthetic",
    }
    info_path = os.path.join(ARTIFACTS_DIR, "model_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print("\n" + "="*55)
    print("MODEL TRAINING COMPLETE")
    print("="*55)
    print(f"  AUC:         {auc:.4f}")
    print(f"  F1:          {f1:.4f}")
    print(f"  Features:    {len(feature_names)}")
    print(f"  Artifacts:   {ARTIFACTS_DIR}/")
    print("="*55)

    return model, feature_names, auc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()
    train_and_save(data_path=args.data_path, random_state=args.random_state)

"""
FastAPI Model Serving Application
===================================
Serves the XGBoost champion model from Project 4.

Endpoints:
  GET  /          — Welcome message
  GET  /health    — Health check
  GET  /model-info — Model metadata
  POST /predict   — Single prediction
  POST /predict-batch — Batch predictions
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")

# ── Load model artifacts on startup ──────────────────────────────────────────
def load_artifacts():
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(ARTIFACTS_DIR, "champion_model.json"))

    with open(os.path.join(ARTIFACTS_DIR, "feature_order.json")) as f:
        feature_order = json.load(f)

    with open(os.path.join(ARTIFACTS_DIR, "encoders.json")) as f:
        encoders = json.load(f)

    with open(os.path.join(ARTIFACTS_DIR, "model_info.json")) as f:
        model_info = json.load(f)

    logger.info(f"Model loaded | AUC: {model_info['auc']} | Features: {len(feature_order)}")
    return model, feature_order, encoders, model_info


model, feature_order, encoders, model_info = load_artifacts()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Bank Marketing Prediction API",
    description="Predicts whether a bank customer will subscribe to a term deposit. XGBoost champion model from MLOps Portfolio Project 4.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Input schema ──────────────────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    age:             int   = Field(..., ge=18, le=95, example=42)
    job:             str   = Field(..., example="admin.")
    marital:         str   = Field(..., example="married")
    education:       str   = Field(..., example="university.degree")
    default:         str   = Field(..., example="no")
    housing:         str   = Field(..., example="yes")
    loan:            str   = Field(..., example="no")
    contact:         str   = Field(..., example="cellular")
    month:           str   = Field(..., example="may")
    day_of_week:     str   = Field(..., example="mon")
    campaign:        int   = Field(..., ge=1, le=50, example=2)
    pdays:           int   = Field(..., example=999)
    previous:        int   = Field(..., ge=0, example=0)
    poutcome:        str   = Field(..., example="nonexistent")
    emp_var_rate:    float = Field(..., example=1.1, alias="emp.var.rate")
    cons_price_idx:  float = Field(..., example=93.994, alias="cons.price.idx")
    cons_conf_idx:   float = Field(..., example=-36.4, alias="cons.conf.idx")
    euribor3m:       float = Field(..., example=4.857)
    nr_employed:     float = Field(..., example=5191.0, alias="nr.employed")

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    prediction:       int
    probability:      float
    label:            str
    confidence:       str
    model_version:    str = "XGBoost-v1"


class BatchRequest(BaseModel):
    customers: List[CustomerFeatures]


class BatchResponse(BaseModel):
    predictions:      List[PredictionResponse]
    total:            int
    positive_count:   int
    positive_rate:    float


# ── Helper ────────────────────────────────────────────────────────────────────
def preprocess_input(customer: CustomerFeatures) -> pd.DataFrame:
    """Convert input to model-ready DataFrame with explicit integer encoding."""
    data = customer.model_dump(by_alias=True)
    row = {}
    for col in feature_order:
        val = data.get(col, 0)
        if isinstance(val, str):
            if col in encoders:
                enc = encoders[col]
                row[col] = int(enc.get(val, -1))
            else:
                row[col] = 0
        else:
            row[col] = float(val) if val is not None else 0.0
    df = pd.DataFrame([row])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df[feature_order]
def make_prediction(customer: CustomerFeatures) -> PredictionResponse:
    df = preprocess_input(customer)
    prob  = float(model.predict_proba(df)[0][1])
    pred  = int(prob >= 0.5)
    label = "Yes — will subscribe" if pred == 1 else "No — will not subscribe"
    confidence = "High" if abs(prob - 0.5) > 0.3 else "Medium" if abs(prob - 0.5) > 0.1 else "Low"
    return PredictionResponse(
        prediction=pred, probability=round(prob, 4),
        label=label, confidence=confidence
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Bank Marketing Prediction API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health")
def health():
    return {
        "status":    "healthy",
        "model":     "XGBoost",
        "auc":       model_info["auc"],
        "features":  len(feature_order),
    }


@app.get("/model-info")
def get_model_info():
    return {
        "model_type":   model_info["model_type"],
        "auc":          model_info["auc"],
        "f1":           model_info["f1"],
        "n_features":   model_info["n_features"],
        "features":     feature_order,
        "baseline_auc": model_info["baseline_auc"],
        "project":      model_info["project"],
        "data_source":  model_info["data_source"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    """Predict term deposit subscription for a single customer."""
    try:
        return make_prediction(customer)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict-batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    """Predict for multiple customers in one request."""
    try:
        predictions = [make_prediction(c) for c in request.customers]
        positive_count = sum(p.prediction for p in predictions)
        return BatchResponse(
            predictions=predictions,
            total=len(predictions),
            positive_count=positive_count,
            positive_rate=round(positive_count / len(predictions), 4) if predictions else 0.0,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)

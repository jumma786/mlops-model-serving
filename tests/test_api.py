"""
Test suite for mlops-model-serving.
Run: pytest tests/ -v --cov=src
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Train model first if artifacts don't exist
if not os.path.exists("artifacts/champion_model.json"):
    from src.model.train import train_and_save
    train_and_save()

from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

SAMPLE_CUSTOMER = {
    "age": 42,
    "job": "admin.",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "mon",
    "campaign": 2,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp.var.rate": 1.1,
    "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4,
    "euribor3m": 4.857,
    "nr.employed": 5191.0
}


class TestHealthEndpoints:
    def test_root(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "message" in r.json()

    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_model_info(self):
        r = client.get("/model-info")
        assert r.status_code == 200
        data = r.json()
        assert "auc" in data
        assert "features" in data
        assert data["auc"] > 0.0


class TestPredictEndpoint:
    def test_predict_returns_200(self):
        r = client.post("/predict", json=SAMPLE_CUSTOMER)
        assert r.status_code == 200

    def test_predict_response_schema(self):
        r = client.post("/predict", json=SAMPLE_CUSTOMER)
        data = r.json()
        assert "prediction" in data
        assert "probability" in data
        assert "label" in data
        assert "confidence" in data

    def test_predict_binary_output(self):
        r = client.post("/predict", json=SAMPLE_CUSTOMER)
        assert r.json()["prediction"] in [0, 1]

    def test_predict_probability_range(self):
        r = client.post("/predict", json=SAMPLE_CUSTOMER)
        prob = r.json()["probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_label_matches_prediction(self):
        r = client.post("/predict", json=SAMPLE_CUSTOMER)
        data = r.json()
        if data["prediction"] == 1:
            assert "Yes" in data["label"]
        else:
            assert "No" in data["label"]


class TestBatchEndpoint:
    def test_batch_predict(self):
        r = client.post("/predict-batch", json={"customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]})
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_positive_rate(self):
        r = client.post("/predict-batch", json={"customers": [SAMPLE_CUSTOMER] * 5})
        data = r.json()
        assert 0.0 <= data["positive_rate"] <= 1.0

    def test_batch_count_matches(self):
        r = client.post("/predict-batch", json={"customers": [SAMPLE_CUSTOMER] * 3})
        data = r.json()
        assert data["positive_count"] == sum(p["prediction"] for p in data["predictions"])

# 🚀 Model Serving — FastAPI + Docker + Cloud Run

![CI](https://github.com/jumma786/mlops-model-serving/actions/workflows/serving.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-champion-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Part of the [MLOps Portfolio Series](https://github.com/jumma786/mlops-portfolio)** — Project 5 of 10  
> Deploys the XGBoost champion model from Project 4 as a production FastAPI service — containerised with Docker, tested with pytest, and ready for Cloud Run deployment.

---

## 📂 Project Resources

| Resource | Link |
|---|---|
| 🚀 FastAPI App | [src/api/app.py](src/api/app.py) |
| 🏋️ Model Trainer | [src/model/train.py](src/model/train.py) |
| 🧪 API Tests | [tests/test_api.py](tests/test_api.py) |
| 🐳 Dockerfile | [Dockerfile](Dockerfile) |
| 🤖 CI/CD Workflow | [.github/workflows/serving.yml](.github/workflows/serving.yml) |
| 📋 Requirements | [requirements.txt](requirements.txt) |

---

## 🎯 What This Project Does

Takes the champion XGBoost model from Project 4 and puts it in front of real users:

1. **Trains champion model** on real UCI Bank Marketing data (41,188 rows)
2. **Serves predictions** via FastAPI REST endpoints
3. **Containerises** with Docker — multi-stage build, non-root user, health check
4. **Tests** all endpoints with pytest (11 passing tests)
5. **CI/CD pipeline** trains → tests → builds Docker → tests container on every push

---

## 📊 Model

| Property | Value |
|---|---|
| Algorithm | XGBoost (tuned with Optuna — Project 4) |
| Dataset | UCI Bank Marketing (41,188 rows, real data) |
| AUC | 0.8176 |
| Baseline AUC | 0.8174 (Project 1 RandomForest) |
| Features | 19 (duration dropped — leakage) |
| Target | Term deposit subscription (binary) |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Welcome message |
| GET | `/health` | Health check + model info |
| GET | `/model-info` | Full model metadata |
| POST | `/predict` | Single customer prediction |
| POST | `/predict-batch` | Batch predictions |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Example Response

```json
{
  "prediction": 0,
  "probability": 0.0823,
  "label": "No — will not subscribe",
  "confidence": "High",
  "model_version": "XGBoost-v1"
}
```

---

## 🏗️ Architecture

```
mlops-model-serving/
├── src/
│   ├── api/
│   │   └── app.py          # FastAPI app — 5 endpoints
│   └── model/
│       └── train.py        # Train + save artifacts
├── tests/
│   └── test_api.py         # 11 pytest tests
├── artifacts/              # model, feature_order, encoders, model_info
├── Dockerfile              # Multi-stage build
├── .github/
│   └── workflows/
│       └── serving.yml     # CI: train → test → docker build → test container
├── requirements.txt
└── Makefile
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/jumma786/mlops-model-serving.git
cd mlops-model-serving
pip install -r requirements.txt

# Train model on real data
python src/model/train.py --data-path data/bank-additional-full.csv

# Run tests
make test

# Start API server
make run
# Open http://localhost:8000/docs
```

---

## 🐳 Docker

```bash
# Build
make docker-build

# Run
make docker-run
# API available at http://localhost:8000
```

---

## ☁️ Cloud Run Deployment

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/mlops-model-serving

# Deploy to Cloud Run
gcloud run deploy mlops-model-serving \
  --image gcr.io/PROJECT_ID/mlops-model-serving \
  --platform managed \
  --region europe-west2 \
  --allow-unauthenticated
```

---

## 🔗 MLOps Portfolio Series

| # | Project | Repo | Status |
|---|---|---|---|
| 1 | Multi-Model Tournament | [mlops-model-tournament](https://github.com/jumma786/mlops-model-tournament) | ✅ |
| 2 | Scheduled Retraining | [mlops-retraining-pipeline](https://github.com/jumma786/mlops-retraining-pipeline) | ✅ |
| 3 | Feature Engineering | [mlops-feature-pipeline](https://github.com/jumma786/mlops-feature-pipeline) | ✅ |
| 4 | Hyperparameter Tuning | [mlops-hyperparameter-tuning](https://github.com/jumma786/mlops-hyperparameter-tuning) | ✅ |
| **5** | **Model Serving** | [mlops-model-serving](https://github.com/jumma786/mlops-model-serving) | ✅ This repo |
| 6 | Feature Store | [mlops-feature-store](https://github.com/jumma786/mlops-feature-store) | ✅ |
| 7 | Model Monitoring | [mlops-model-monitoring](https://github.com/jumma786/mlops-model-monitoring) | ✅ |
| 8 | A/B Testing | [mlops-ab-testing](https://github.com/jumma786/mlops-ab-testing) | ✅ |
| 9 | Airflow Pipeline | [mlops-airflow-pipeline](https://github.com/jumma786/mlops-airflow-pipeline) | ✅ |
| 10 | Kubernetes Platform | [mlops-k8s-platform](https://github.com/jumma786/mlops-k8s-platform) | ✅ |

---

## 📝 Key MLOps Concepts Demonstrated

- **REST API serving** — FastAPI with Pydantic validation
- **Containerisation** — Docker multi-stage build, non-root user
- **CI/CD for serving** — train → test → build → test container
- **Input validation** — Pydantic schemas reject invalid inputs
- **Health checks** — Docker HEALTHCHECK + /health endpoint
- **Batch inference** — /predict-batch for high-throughput use cases

---

## 👤 Author

**Jumma Mohammad Teli** — Data Analyst & ML Engineer  
📍 Birmingham, UK  
📧 [jummamohammad477@gmail.com](mailto:jummamohammad477@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/jumma-mohammad) | [GitHub](https://github.com/jumma786)

---

*Project 5 of 10 — MLOps Portfolio Series. Builds on Project 4 (Hyperparameter Tuning) by deploying the champion model as a production API.*

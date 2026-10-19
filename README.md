# 🎬 CineMatch MLOps — Movie Recommendation System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-enabled-2496ED.svg)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-ready-326CE5.svg)](https://kubernetes.io/)
[![Prometheus](https://img.shields.io/badge/Prometheus-instrumented-E6522C.svg)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-observable-F46800.svg)](https://grafana.com/)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-0194E2.svg)](https://mlflow.org/)

An end-to-end production-grade **MLOps Movie Recommendation System** built for a 4-member engineering team. This system automatically trains, evaluates, packages, deploys, monitors, and retrains recommendation models based on real-time feedback and data drift detection.

---

## 🏛️ Architecture Overview

```
                                 [ User / Web UI ]
                                        │
                                        ▼
                             [ NGINX Reverse Proxy ]
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
       [ FastAPI Serving API ]                       [ Prometheus Monitoring ]
        • /recommend/{user_id}                         • Request latency & rate
        • /ratings (Ingestion)                         • NDCG@10 & Recall@10
        • /metrics & /health                           • Drift detection score
                 │                                             │
                 ▼                                             ▼
       [ Production Model ]                           [ Grafana Dashboard ]
        • Collaborative Filtering (SVD)                • Real-time visualization
        • Content-Based (Cosine Genre)
        • Hybrid Recommender (Adaptive)
                 │
                 ▼
       [ PostgreSQL Database ]
        • Movie catalog & demographics
        • Ratings & feedback buffer
                 │
                 ▼
       [ Automated Retraining Loop ]
        • Threshold: 50+ new ratings or KS-drift
        • Quality Gate: NDCG@10 >= 0.15 & no regression
        • MLflow tracking & Model Registry promotion
```

---

## 👥 4-Member Team Ownership

| Team Member | Area | Core Responsibilities |
| :--- | :--- | :--- |
| **Member 1** | **Recommendation Algorithms** | MovieLens data pipeline, feature engineering, SVD collaborative filtering, content-based model, and hybrid weighting logic. |
| **Member 2** | **MLOps & Experiment Tracking**| MLflow tracking server, parameter/metric logging, model registry versioning, and automated retrain quality gates. |
| **Member 3** | **API & Frontend** | FastAPI recommendation endpoints, SQLAlchemy/PostgreSQL integration, rating ingestion, and modern responsive web UI. |
| **Member 4** | **DevOps & Monitoring** | Docker containerization, Jenkins CI/CD pipeline, Kubernetes manifests (Deployment/HPA/Ingress), Prometheus metrics, and Grafana dashboard. |

---

## 🚀 Quickstart Guide

### Option 1: Standalone Local Run (Zero Docker Required)

1. **Clone & install dependencies:**
   ```bash
   git clone <repo-url>
   cd MLOPS_INNOVATIVE
   pip install -r requirements.txt
   ```

2. **Run the master setup script:**
   ```bash
   # Seeds database, trains models, evaluates metrics, and starts the API + UI
   python run_project.py all
   ```

3. **Open in your browser:**
   - **Interactive Web App**: [http://localhost:8000](http://localhost:8000)
   - **Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Prometheus Metrics**: [http://localhost:8000/metrics](http://localhost:8000/metrics)

---

### Option 2: Full Production Docker Stack

Run all 6 orchestrated containers (PostgreSQL, MLflow, FastAPI API, NGINX, Prometheus, Grafana) with a single command:

```bash
docker-compose up --build -d
```

| Service | Port | Description |
| :--- | :--- | :--- |
| **NGINX / Web App** | `http://localhost:80` | Main entry point & Web UI |
| **FastAPI Backend** | `http://localhost:8000` | Recommendation API & Docs |
| **MLflow Server** | `http://localhost:5000` | Experiment tracking & Model Registry |
| **Prometheus** | `http://localhost:9090` | Time-series metrics collection |
| **Grafana** | `http://localhost:3001` | Observability Dashboard (`admin` / `admin`) |
| **PostgreSQL** | `localhost:5432` | Application interaction database |

---

## 🧪 Testing & Quality Gates

Run the comprehensive automated test suite (data validation, model logic, API routes, and drift detectors):

```bash
pytest tests/ -v
```

---

## 🔄 Automated Retraining Workflow

1. Users rate movies in the Web UI or via `POST /ratings`.
2. Each rating is saved to the database with `is_new = 1`.
3. When new ratings exceed the threshold (`50`), background retraining is triggered.
4. **Quality Gate Evaluation**:
   - `NDCG@10` must exceed minimum baseline (`0.15`).
   - Candidate performance must not regress by > 5% compared to the active production model.
5. If passed, the candidate model is promoted to production and Prometheus gauges are updated.

---

## ☸️ Kubernetes Deployment

Deploy the system to a local (Minikube / kind) or cloud Kubernetes cluster:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/nginx-ingress.yaml

# Verify deployment status
kubectl get pods -n movie-rec
kubectl rollout status deployment/recommender-api -n movie-rec
```

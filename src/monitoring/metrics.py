"""
Prometheus Metrics Instrumentation for FastAPI and MLOps Pipeline
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Request metrics
REQUEST_COUNT = Counter(
    "recommendation_requests_total",
    "Total recommendation requests received",
    ["endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "recommendation_request_latency_seconds",
    "Time spent processing recommendation request",
    ["endpoint"]
)

MODEL_INFERENCE_TIME = Histogram(
    "model_inference_seconds",
    "Time taken purely by the recommender model prediction",
    ["model_type"]
)

# Business / MLOps metrics
RECOMMENDATIONS_SERVED = Counter(
    "recommendations_served_total",
    "Total count of movie recommendations returned to users"
)

RATINGS_INGESTED = Counter(
    "ratings_ingested_total",
    "Total new user ratings submitted"
)

ACTIVE_MODEL_VERSION = Gauge(
    "active_model_version",
    "Currently deployed model version identifier",
    ["model_name"]
)

MODEL_NDCG_SCORE = Gauge(
    "model_ndcg_at_10",
    "Latest evaluated NDCG@10 of the active production model"
)

MODEL_RECALL_SCORE = Gauge(
    "model_recall_at_10",
    "Latest evaluated Recall@10 of the active production model"
)

DATA_DRIFT_SCORE = Gauge(
    "data_drift_p_value",
    "P-value from KS-test for rating distribution drift (lower means drift)"
)

DRIFT_ALERT_TRIGGERED = Gauge(
    "drift_alert_triggered",
    "1 if data drift exceeds threshold, 0 otherwise"
)

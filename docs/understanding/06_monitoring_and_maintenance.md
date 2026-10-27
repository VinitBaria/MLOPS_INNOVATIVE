# Monitoring and maintenance

## Metrics, logs, and dashboards

`src/monitoring/metrics.py` defines Prometheus metrics:

| Metric | Type | Updated by |
|---|---|---|
| `recommendation_requests_total{endpoint,status}` | Counter | Recommendation handler increments received/success (not all endpoints). |
| `recommendation_request_latency_seconds{endpoint}` | Histogram | Recommendation handler. |
| `model_inference_seconds{model_type}` | Histogram | Recommendation model call. |
| `recommendations_served_total` | Counter | Recommendation response creation. |
| `ratings_ingested_total` | Counter | Successful rating upsert. |
| `active_model_version{model_name}` | Gauge | Model loader sets a constant value of 1; it does not encode the version number. |
| `model_ndcg_at_10`, `model_recall_at_10` | Gauges | Model load and retraining promotion path. |
| `data_drift_p_value`, `drift_alert_triggered` | Gauges | Drift helper if invoked. |

Prometheus scrapes API `/metrics` and itself every five seconds (`docker/prometheus/prometheus.yml`). Grafana provisions a Prometheus datasource and dashboard with request rate, P95 latency, NDCG@10, Recall@10, and drift alert query (`docker/grafana/dashboards/dashboard.json`). This is visualization config; no alert rules or notification routes are included. The README says request rate and drift score are observed, but the implemented `REQUEST_COUNT` only tracks recommendation handler stages.

Console `print` statements are used for training and errors; no structured logging setup is present. Recommendation request logs are persisted in SQL (`src/database/models.py::RecommendationLog`, `src/api/main.py::get_recommendations`), but the code does not write `DriftLog`. No trace correlation, centralized logging config, dashboard screenshots, or retention configuration is supplied.

## Drift and performance monitoring

`src/monitoring/drift.py::detect_rating_drift` compares baseline and recent rating distributions using:

- Two-sample Kolmogorov–Smirnov test (KS), with drift when `p_value < 0.05` by default.
- Population Stability Index (PSI), with drift when PSI ≥ 0.15 by default. The helper comment labels <0.1 no significant change, 0.1–0.2 moderate, ≥0.2 significant.
- Fewer than 20 recent ratings returns `INSUFFICIENT_DATA` and does not update gauges.

This helper measures input rating-value distribution shift, not model concept/performance drift. It is not called from API, training, or retraining code. `.env.example`'s `DRIFT_THRESHOLD=0.1` is not read by the helper. No code queries recent ratings into baseline/recent arrays or persists results.

NDCG/Recall gauges are offline evaluation results loaded from metadata or set after retraining. There is no online relevance feedback join, A/B test, click-through measurement, delayed label monitor, or retraining schedule. Performance counters/histograms are online service metrics; model quality gauges are offline signals.

## Retraining, promotion, rollback

`src/training/retrain.py::run_automated_retraining` counts `Rating.is_new`, loads prior NDCG/version metadata, trains, checks absolute NDCG threshold 0.15 and regression floor 5%, then returns promoted/rejected status. `force=True` overrides failed gates. However:

1. `train_pipeline` saves over the same local production path **before** the decision.
2. A rejected candidate is not restored from a prior artifact snapshot.
3. A promoted candidate is not reflected into already-running API globals.
4. The function does not update `metadata.json` with a promoted model version or invoke MLflow `promote_model`.
5. It marks all `is_new` DB ratings as processed after a pass even though those SQL ratings were not the training pipeline's input.

Consequently there is no dependable rollback or serving version switch implemented. Manual `POST /model/reload` can reload disk artifacts, but has no authentication and does not validate artifact compatibility (`src/api/main.py::reload_model`).

## Security and configuration

- `.env.example` contains development-looking fixed passwords and a `SECRET_KEY`; Compose and Kubernetes deployment also embed default DB credentials. These values are examples/config, not a secrets manager integration. Replace and rotate them for a real deployment.
- `src/database/db.py` loads `.env` and supports environment overrides. `.env` is ignored by `.gitignore`; `.env.example` is tracked.
- No authentication, authorization, rate limit at the app itself, CSRF protections, or role checks are implemented in API handlers. NGINX applies a per-IP request limit in Compose config, but K8s ingress has a different annotation.
- API CORS uses `allow_origins=["*"]`, `allow_credentials=True`; `GET /metrics`, `/model/retrain`, and `/model/reload` are not access-controlled.
- Joblib artifacts are deserialized from local storage; only load trusted files. The app binds `0.0.0.0` in Docker and local runner.
- TLS, secret rotation, DB backups/restore procedure, artifact signing, vulnerability scanning, and dependency lockfiles are absent from the repository.

These are observations from `.env.example`, Docker/Kubernetes configs, `src/api/main.py`, `src/database/db.py`, and `.gitignore`; operational risk depends on actual exposure and overrides, which are not documented.

# Pipeline orchestration and CI/CD

## Orchestration

No Airflow, Prefect, Kubeflow, or other scheduled DAG definition is present. The actual local orchestrator is `run_project.py::main`, a sequential CLI with actions `setup`, `train`, `test`, `serve`, and `all`. `all` seeds, trains, then starts the web server; it does not run tests. `train_pipeline` performs download → validate → preprocess → fit → evaluate → local save → optionally MLflow logging. No schedule is defined in the Python source.

Ratings can schedule a background retrain in the API once the pending count reaches a threshold (`src/api/main.py::submit_rating`); manual `POST /model/retrain` also schedules it. These are request-triggered FastAPI background tasks, not durable queues or a scheduler. `.env.example` declares `RETRAIN_SCHEDULE_HOURS=24`, but no scheduler reads this setting. `src/monitoring/drift.py::detect_rating_drift` is not called by the retraining trigger.

| Stage | Input → output | Trigger/schedule |
|---|---|---|
| Setup/seed | MovieLens data → SQL `movies`, `users`, baseline `ratings` | `run_project.py setup` or `all`; only seeds if no movie rows exist. |
| Train | MovieLens files → fitted model artifacts, preprocessor, metadata, optional MLflow runs | CLI/manual; no schedule. |
| API inference | HTTP request and DB state → recommendations/ratings/metrics | HTTP. |
| Feedback retrain | `ratings.is_new` count → call `train_pipeline`, then quality decision | Rating threshold or manual endpoint; background task is in-process. |
| Drift check | baseline/recent rating sequences → KS/PSI report and gauges | No trigger found; helper must be invoked by external code. |

## GitHub Actions

`.github/workflows/ci-cd.yaml` triggers on pushes and pull requests targeting `main` or `master`.

1. `test-and-validate` checks out source, sets Python 3.11, caches pip, installs requirements, and runs `pytest tests/ --disable-warnings -v`.
2. “Quality Gates” runs only `python -c "print('Quality Gate Validation Passed!')"`; this is not a model-quality check.
3. `build-and-package` depends on tests and runs only if `github.ref == 'refs/heads/main'`; it configures Buildx and runs `docker build -t movie-recommender-api:latest .`.
4. There is no push to a registry, Kubernetes deployment, or smoke test in this workflow.

The Docker build command is expected to encounter the missing `frontend/` directory referenced by `Dockerfile`; this was not run here.

## Jenkins pipeline

`Jenkinsfile` describes checkout, Python venv install, syntax compilation of two files, pytest, metadata threshold check, Docker build/tag, publish, K8s apply, rollout, smoke-test, and workspace cleanup.

Important discrepancies and gaps:

- The metadata gate reads `models/artifacts/metadata.json` only if it exists; despite echoing NDCG ≥ 0.15, it fails only below 0.10. A missing metadata file passes.
- The registry push commands are commented out; “Publish Docker Image” only echoes.
- Deploy applies Namespace, PostgreSQL, API, and Ingress but not ConfigMap or HPA.
- Deployment expects `localhost:8000` for smoke tests and a K8s endpoint may not be reachable there.
- Failure message says rollback, but there is no rollback command.
- `cleanWs()` deletes the Jenkins workspace after the run (pipeline behavior, not repository files).

Jenkins was not run; it needs a Jenkins environment, installed plugins/tools, Docker daemon, and Kubernetes access.

## Containerization and deployment

`Dockerfile` has a build stage installing dependencies and a runtime stage copying Python dependencies/source. It exposes 8000 and launches Uvicorn. The build stage installs `curl`, but runtime stage does not explicitly install it even though healthcheck invokes curl. `frontend/` is also copied but absent from the repository. The image contains source but no model/preprocessor artifacts.

`docker-compose.yml` defines PostgreSQL 15, MLflow 2.9.2, API, NGINX, Prometheus 2.48.0, Grafana 10.2.0. API depends on Postgres health and MLflow start; it mounts `./models` and `./data`, so local model artifacts still have to exist. Compose sets retraining threshold to 50; `.env.example` instead shows 5000, and the API default is 50. Compose configuration parsing succeeded locally; Compose warned that the top-level `version` field is obsolete. No services were started.

Kubernetes config under `k8s/` provides a three-replica API deployment, rolling update, CPU/memory requests/limits, readiness/liveness checks, a ClusterIP service, CPU/memory HPA 2–10 replicas, NGINX ingress, and one PostgreSQL pod with a 5 Gi PVC. The config map has no password; deployment embeds a literal password. Model artifact and preprocessor volumes are absent. MLflow, Prometheus, and Grafana are not deployed by the checked-in K8s YAML.

**Why:** containers standardize runtime and Kubernetes replicas/HPA/probes describe availability/scaling. **my inference** from deployment config; the repo contains no ADR or rationale. Alternatives include Compose alone or managed platform services; simpler alternatives reduce operations work but may offer fewer scheduling/scaling controls.

## Testing strategy

| Type | Files/tests | Current coverage |
|---|---|---|
| Model unit tests | `tests/test_models.py` | Fit/predict/recommend smoke checks for CF, content, and hybrid on a tiny matrix. Does not test persistence, cold start, score quality, or real MovieLens input. |
| Data unit tests | `tests/test_data.py` | Valid/invalid ratings and single-rating validation. No downloader, movie validation, preprocessing/split, leakage, or database seeding integration test. |
| Drift unit tests | `tests/test_drift.py` | PSI on synthetic distributions and drift decision. Some test data uses unseeded random numbers. |
| API tests | `tests/test_api.py` | TestClient health, metrics, model info, recommendation shape, and rating status 200/404. They do not seed fixtures or assert recommendation quality, retrain, similarity, search, or auth. Importing `src.api.main` constructs DB engine/schema. |
| CI syntax | Jenkinsfile | `py_compile` only for `src/api/main.py` and `src/models/hybrid.py`; no lint/type checker command. |
| CI model gate | Jenkinsfile | Optional metadata check with an actual 0.10 cutoff and missing file pass-through, not a live re-evaluation. |

Test commands are configured, but test execution requires dependencies and a safe DB target; see [08_how_to_run.md](./08_how_to_run.md).

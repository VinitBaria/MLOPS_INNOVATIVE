# User stories and implemented flows

“Implemented” here means a code path exists in this checkout; it does not imply production hardening or that external prerequisites are installed. “Partial” means the flow is exposed but incomplete, has a documented code gap, or depends on missing artifacts. Code citations use `path::function_or_class`.

## End user

### Get personalized recommendations — **Partial**

> As an end user, I want movie recommendations, so that I can discover films I may enjoy.

Flow: `GET /recommend/{user_id}?limit=n` → `src.api.main::get_recommendations` loads the user's watched movie IDs from `Rating`, maps known IDs with the saved `DataPreprocessor`, calls `HybridRecommender.recommend`, maps internal movie indices to database `Movie` rows, and returns `RecommendationResponse` containing `MovieItem`s. The route logs recommendation output and latency to `RecommendationLog`. If artifacts are missing, inference errors, or no movies map back, it returns a database-order fallback marked `popularity_fallback` with a hard-coded score of `4.0`; it does not calculate popularity. Schemas: `src/api/schemas.py::RecommendationResponse`, `MovieItem`. Database: `src/database/models.py::Rating`, `Movie`, `RecommendationLog`.

Why design choice — **my inference:** combining interaction-based and genre-based rankings aims to give both personalization and a content signal. The code has no explicit product reason or comparison to alternatives. Limitation: fallback ordering is not popularity-ranked, and startup needs local model artifacts that may not exist.

### Search / browse the catalog — **Implemented in API**

> As an end user, I want to search and filter movies, so that I can find a title or genre.

Flow: `GET /movies?query=&genre=&skip=&limit=` → `src.api.main::list_movies` applies case-insensitive title/genre SQL filters and pagination, then returns `List[MovieItem]`. `skip` and `limit` have no explicit range validation in the route.

### Find similar movies — **Implemented with fallback**

> As an end user, I want similar-movie suggestions, so that I can explore related titles.

Flow: `GET /movies/{movie_id}/similar?limit=` → `src.api.main::get_similar_movies` checks catalog existence, maps movie ID to model index and delegates to `HybridRecommender.get_similar_movies` / `ContentBasedModel.get_similar_movies`. If model output is unavailable or throws, it filters by the first genre and labels each fallback similarity `0.85`. The similarity endpoint's broad exception handler hides inference causes.

### Rate a movie — **Partial**

> As an end user, I want to submit or update a rating, so that my feedback is stored and may improve recommendations.

Flow: `POST /ratings` with `RatingCreate` → `src.api.main::submit_rating` validates numeric bounds via Pydantic, checks movie existence, upserts a `Rating`, counts `is_new == 1`, and schedules `run_automated_retraining` when the count reaches `RETRAIN_THRESHOLD_NEW_RATINGS` (default 50). It returns `RatingResponse`. It does **not** create an unknown `User` row, even though `Rating.user_id` is a foreign key to `users.user_id`; PostgreSQL may reject a new-user rating. Also, feedback is saved to the database but is not included in `train_pipeline`'s MovieLens-file training input. References: `src/api/schemas.py::RatingCreate`, `src/database/models.py::Rating`, `src/training/train.py::train_pipeline`.

## Data scientist / ML engineer

### Train and compare recommenders — **Implemented**

> As a data scientist, I want to train and compare candidate recommenders, so that I can select a model by measured offline quality.

Flow: `python run_project.py train` → `src/training/train.py::train_pipeline` downloads/loads MovieLens 100K, validates, calls `DataPreprocessor.fit_transform`, fits the three model classes, computes evaluation metrics, selects the maximum `ndcg_at_10`, and saves per-model plus `production` artifacts and JSON metadata. The CLI exposes `n_factors`, weights, split size, and MLflow flag when invoking `train.py` directly; `run_project.py` uses fixed `n_factors=20` and disables MLflow.

### Inspect tracked experiments — **Partial**

> As an ML engineer, I want to track parameters, metrics, and model artifacts, so that I can compare training runs.

`src/training/mlflow_utils.py::setup_mlflow` chooses an experiment and `log_training_run` writes params/metrics/tags/artifacts. `train_pipeline(use_mlflow=True)` invokes these helpers. The default CLI runner passes `use_mlflow=False`; local MLflow endpoint availability is not required for that path. Tracking is optional and its default URI depends on environment; no experiment comparison UI or automatic candidate-to-serving resolution is wired in code.

## MLOps / DevOps engineer

### Build and deploy the API — **Partial**

> As an MLOps engineer, I want a container and deployment manifests, so that I can run the API consistently.

GitHub Actions runs tests and, on `main`, a Docker image build (`.github/workflows/ci-cd.yaml`). `Dockerfile` packages Uvicorn and the source; `docker-compose.yml` describes six services. Kubernetes YAML describes API/PostgreSQL deployment, ingress, probes, and HPA (`k8s/*.yaml`). The Docker build references missing `frontend/`; the Kubernetes image is a local tag and no image publication is configured in the GitHub workflow. These are descriptions/configuration, not evidence of a successful deployment.

### Monitor API / model metrics — **Partial**

> As an MLOps engineer, I want to inspect service and model signals, so that I can detect degradation.

API counters, histograms, and gauges are exposed at `/metrics`; Prometheus scrapes `api:8000`; Grafana provisions a dashboard. The standalone `src/monitoring/drift.py::detect_rating_drift` calculates KS and PSI for rating samples but no scheduler or API path invokes it. Thus request/model metrics are wired; automated drift monitoring and alerts are not.

### Retrain and roll back — **Partial / rollback not implemented**

> As an MLOps engineer, I want retraining to promote only better candidates and restore a previous version after failure, so that releases are safe.

Threshold-driven `src.api.main::submit_rating` and manual `POST /model/retrain` enqueue `src.training.retrain::run_automated_retraining`; it compares candidate NDCG@10 with a floor. The retraining function calls `train_pipeline` before the quality gate, and that training function overwrites local `production` artifacts. If the gate rejects, files are not restored. It also does not update in-memory `MODEL` or call the existing MLflow promotion helper. Therefore the quality check exists, but reliable promotion/rollback does not.

## Admin / maintainer

### Inspect health and model info — **Implemented, limited**

> As a maintainer, I want to see whether the DB/model are ready and which model metadata is loaded, so that I can diagnose basic service state.

`GET /health` → `src.api.main::health_check` returns DB connectivity, `MODEL is not None`, and a metadata version; `GET /model/info` → `src.api.main::model_info` returns selected model metadata and offline metrics. No authentication or operational role system is present.

### Manage users, data, or access — **Not implemented**

There is no admin UI, user-management endpoint, authorization layer, or credential rotation flow in the checked-in API (`src/api/main.py`, `src/api/schemas.py`). The configuration includes a `SECRET_KEY`, but no code uses it.

## Who actually calls what

The only checked-in executable orchestrator is `run_project.py`; its `setup` action seeds the DB, `train` trains from MovieLens files, `test` invokes pytest, `serve` starts Uvicorn, and `all` sequentially does setup → train → serve. It does not run tests in `all`, and serving blocks the process. Source: `run_project.py::main`.

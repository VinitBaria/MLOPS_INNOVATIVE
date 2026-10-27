# Serving and API

## Runtime

`src/api/main.py` constructs a FastAPI application, creates ORM tables at module import (`Base.metadata.create_all`), and calls `load_model_and_preprocessor` at application startup. It loads `DataPreprocessor` from `data/processed/preprocessor.joblib` and expects `HybridRecommender` artifacts under `models/artifacts/production`. The objects live in module globals, so inference reuses the in-memory objects between requests. On loading errors, the handler prints a notice and leaves fallback mode available.

**Why:** caching a loaded object avoids deserializing it on each request. **my inference** based on global model state and startup loader; the repository does not explain this choice. Alternatives include an inference service or process-local cache with controlled refresh; they bring more infrastructure or consistency concerns.

## Endpoint catalog

Request/response models are defined in `src/api/schemas.py`; endpoint behavior is in `src/api/main.py`.

| Method + path | Query/body | Response | Implementation |
|---|---|---|---|
| `GET /health` | None | `HealthResponse`: status, UTC timestamp, `model_loaded`, model version, `db_connected` | `health_check`; attempts a `Movie` query and calls status healthy only if DB query works and model is loaded. HTTP response can still be 200 when status is `"degraded"`. |
| `GET /metrics` | None | Prometheus text exposition | `metrics`; emits default registry via `generate_latest`. |
| `GET /model/info` | None | `ModelInfoResponse`: name/version, selected type, NDCG@10, Recall@10, RMSE, training timestamp | `model_info`; reads in-memory JSON metadata. |
| `GET /recommend/{user_id}` | `limit` default 10, bounds 1–50 | `RecommendationResponse`: user ID, `MovieItem[]`, model version/type, latency ms | `get_recommendations`. `MovieItem` includes movie ID, title, genre string, optional predicted rating/similarity. |
| `GET /movies` | Optional `query`, `genre`; `skip=0`, `limit=20` | `MovieItem[]` | `list_movies`; case-insensitive DB title/genre match. No declared bounds on pagination. |
| `GET /movies/{movie_id}/similar` | `limit=6` | `MovieItem[]` with optional similarity score | `get_similar_movies`; 404 when catalog ID is absent. |
| `POST /ratings` | JSON `RatingCreate`: positive user/movie IDs, rating 1–5 | `RatingResponse`: status/message, submitted IDs/rating, pending count, retraining flag | `submit_rating`; 404 if movie does not exist. |
| `POST /model/retrain` | None | `{status, message}` | `trigger_retrain`; schedules background retraining with `force=True`. |
| `POST /model/reload` | None | `{status, message}` | `reload_model`; reloads local artifacts into process globals. |

`RatingCreate` validation occurs before handler execution (`src/api/schemas.py::RatingCreate`). Movie existence is checked, but user existence is not. Response schemas do not report exceptions from background training. The default FastAPI `/docs` and `/openapi.json` routes are available unless disabled (not disabled in `FastAPI(...)`).

## Recommendation request lifecycle

1. FastAPI validates path/query (`get_recommendations`).
2. SQLAlchemy loads ratings for `user_id`; the resulting movie IDs are excluded (`src/database/db.py::get_db`, `src/database/models.py::Rating`).
3. If model and preprocessor globals exist, raw user/movie IDs are encoded. Unknown user gets sentinel index `999999`; unknown catalog movies are not added to exclusions.
4. The handler calls `HybridRecommender.recommend(user_idx, n=limit, exclude_watched=...)`; inference latency is observed.
5. Model indices are decoded and individually queried against `Movie`. Only rows found become response items.
6. If there are no items, database fallback takes up to `limit` movies excluding watched IDs, but does not order by rating or interaction popularity.
7. Request metrics and recommendation counters are updated; a best-effort recommendation log is committed.
8. The response includes a model version default, model type (`hybrid` or `popularity_fallback`), and measured latency.

## Caching, batching, and latency

There is no batching, asynchronous model inference, database connection pooling policy beyond SQLAlchemy's default pool, response cache, or explicit model cache invalidation. The API makes per-movie catalog queries after model recommendation (`get_recommendations`), potentially an N+1 query pattern. Cold start does not itself request a popularity-ranked set. No service SLO or latency budget is recorded.

The code's global model is process-local; multiple K8s replicas each load a local artifact copy. No manifest volume mounts `models/artifacts` or `data/processed`, so the deployed pod may start without artifacts. `load_model_and_preprocessor` also always uses `HybridRecommender.load`, while training copies whichever model had the best NDCG into `production`. If the best model is not hybrid, load is incompatible with the production artifact files. References: `src/api/main.py::load_model_and_preprocessor`, `src/training/train.py::train_pipeline`, `k8s/api-deployment.yaml`.

## Frontend

No frontend source or build files are tracked. `src/api/main.py` mounts `frontend/build` only if that directory exists; the Dockerfile unconditionally executes `COPY frontend/ /app/frontend/`, but the directory is absent in the inventory. Thus the repository does not provide a verifiable UI flow or show how a browser calls the API. README's “Interactive Web App” is not supported by a checked-in frontend implementation. `data/top_posters.json` contains title-keyed metadata but is not read by checked-in Python code.

## API hardening

The application configures CORS with wildcard origins, methods, and headers while allowing credentials (`src/api/main.py` application setup). No authentication/authorization protects model-management endpoints or metrics. The behavior is confirmed by code; whether this is acceptable depends on deployment context, which the repository does not state.

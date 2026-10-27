# Glossary

Definitions below are general ML/MLOps terms; project-specific usage is cited in each entry.

| Term | Meaning in this project |
|---|---|
| **API (Application Programming Interface)** | HTTP interface for recommendations, catalog, ratings, status, metrics, and model operations (`src/api/main.py`). |
| **Artifact** | Saved output of training, such as model state, encoder state, or metadata (`src/training/train.py`, `src/models/*.py`). |
| **ASGI** | Asynchronous Server Gateway Interface, the Python server/application interface used by Uvicorn/FastAPI (`Dockerfile`, `src/api/main.py`). |
| **Baseline** | Reference model/data/metric used for comparison; retraining uses a stored NDCG value and drift code expects baseline ratings (`src/training/retrain.py`, `src/monitoring/drift.py`). |
| **Batching** | Combining multiple inference inputs into one operation; not implemented in API. |
| **CI (Continuous Integration)** | Automated validation on source changes; implemented in `.github/workflows/ci-cd.yaml` and described in `Jenkinsfile`. |
| **CD (Continuous Delivery/Deployment)** | Automated packaging/release/deployment; GitHub Actions only builds, Jenkins describes a deployment flow (`.github/workflows/ci-cd.yaml`, `Jenkinsfile`). |
| **Cold start** | Sparse/no history for a user or item; hybrid/content fallbacks are in `src/models/hybrid.py` and `content_based.py`. |
| **Collaborative filtering (CF)** | Recommendations based on interaction patterns among users and items (`src/models/collaborative.py`). |
| **Content-based filtering** | Recommending items based on item features and a user's past-item profile (`src/models/content_based.py`). |
| **Cosine similarity** | Normalized vector similarity based on angle; used for genre vectors and user profiles (`src/models/content_based.py`). |
| **Counter** | Prometheus metric that only increases, such as rating submissions (`src/monitoring/metrics.py`). |
| **CSR matrix** | Compressed Sparse Row matrix representation; used for user/item interactions (`src/data/preprocess.py`). |
| **Data drift** | Change in input data distribution; the rating distribution detector uses KS and PSI (`src/monitoring/drift.py`). |
| **Data leakage** | Evaluation/training information crossing a split boundary and making results unrealistically optimistic; `DataPreprocessor` documents temporal split as leakage prevention. |
| **Deployment** | Running an application in a target environment; configs describe Compose and Kubernetes (`docker-compose.yml`, `k8s/`). |
| **Embedding / latent factor** | Dense numeric vector representing an entity; CF SVD yields user/item factor matrices (`src/models/collaborative.py`). |
| **Endpoint** | HTTP method/path handled by an API function, such as `GET /recommend/{user_id}` (`src/api/main.py`). |
| **Experiment tracking** | Recording run parameters, metrics, tags, artifacts; optional MLflow helpers provide this (`src/training/mlflow_utils.py`). |
| **Feature engineering** | Transforming raw data into model inputs; here ID encoding and genre-vector extraction (`src/data/preprocess.py`). |
| **Feature store** | Managed, versioned store for reusable offline/online ML features; none is present. |
| **Gauge** | Prometheus metric that can go up/down, e.g. the active model marker (`src/monitoring/metrics.py`). |
| **HPA (Horizontal Pod Autoscaler)** | Kubernetes controller scaling replica count based on resource metrics (`k8s/hpa.yaml`). |
| **Hybrid recommender** | Combination of collaborative and content-based scores (`src/models/hybrid.py`). |
| **Inference** | Applying a trained model to a request; `get_recommendations` calls `MODEL.recommend` (`src/api/main.py`). |
| **Integration test** | Test crossing component boundaries; `tests/test_api.py` uses FastAPI `TestClient`. |
| **Joblib** | Python object serialization library used for local model/preprocessor artifacts (`src/models/*.py`, `src/data/preprocess.py`). |
| **KS test (Kolmogorov–Smirnov)** | Two-sample statistical test for distribution differences (`src/monitoring/drift.py::detect_rating_drift`). |
| **Label encoding** | Mapping category identifiers to integer indices; `LabelEncoder` used for users and movies (`src/data/preprocess.py`). |
| **MAE (Mean Absolute Error)** | Average absolute difference between predicted and actual ratings (`src/models/evaluate.py`). |
| **Matrix factorization** | Approximation of a user-item matrix as low-rank factors (`src/models/collaborative.py`). |
| **Metadata** | Descriptive values associated with a movie/model; e.g. genre, title, hyperparameters, evaluation metrics (`src/database/models.py`, training JSON). |
| **Metric** | Numeric measurement used to evaluate model or service behavior (`src/models/evaluate.py`, `src/monitoring/metrics.py`). |
| **MLflow Model Registry** | MLflow facility for named/versioned registered models and stages; utilities exist but API does not load from it (`src/training/mlflow_utils.py`). |
| **NDCG@K** | Normalized Discounted Cumulative Gain at the top K ranks (`src/models/evaluate.py::_ndcg_at_k`). |
| **ORM (Object-Relational Mapper)** | Maps Python classes to relational database tables; SQLAlchemy models are in `src/database/models.py`. |
| **P95 latency** | 95th percentile response time; Grafana estimates it from Prometheus histogram buckets (`docker/grafana/dashboards/dashboard.json`). |
| **Precision@K** | Fraction of top-K recommendations that are relevant (`src/models/evaluate.py`). |
| **PSI (Population Stability Index)** | Binned expected/actual distribution divergence indicator (`src/monitoring/drift.py::calculate_psi`). |
| **Prometheus** | Metrics collection and time-series query server configured in `docker/prometheus/prometheus.yml`. |
| **Recall@K** | Fraction of relevant items retrieved among top K (`src/models/evaluate.py`). |
| **Registry** | Catalog tracking model versions and lifecycle states; MLflow registration is optional (`src/training/mlflow_utils.py`). |
| **Relevance threshold** | Rating cutoff that determines whether a held-out item counts as relevant; constructor parameter exists but is unused (`src/models/evaluate.py::ModelEvaluator`). |
| **RMSE (Root Mean Squared Error)** | Square root of mean squared rating prediction error (`src/models/evaluate.py`). |
| **Rollback** | Restoring the previously deployed version after a failed release; no artifact rollback is implemented (`src/training/retrain.py`). |
| **SVD (Singular Value Decomposition)** | Matrix decomposition into singular vectors/values; truncated SciPy `svds` is used (`src/models/collaborative.py`). |
| **Temporal split** | Train/test split in chronological order (`src/data/preprocess.py::DataPreprocessor._temporal_split`). |
| **Test coverage** | Which code paths and behaviors are exercised by tests; current suite is in `tests/`. |
| **Two-tower model** | Retrieval architecture with separate user/item neural encoders; not implemented in this repository. |
| **Uvicorn** | ASGI server used to run the FastAPI application (`run_project.py`, `Dockerfile`). |
| **Validation** | Checks that input satisfies expected schema/ranges/quality rules (`src/data/validate.py`, Pydantic schemas). |
| **Vector** | Ordered numeric feature representation; movie genres and user genre profiles are vectors (`src/models/content_based.py`). |
| **Versioning** | Keeping identifiable immutable revisions of data/models/config; only metadata and MLflow registration helpers exist, with no full data versioning pipeline (`src/training/train.py`, `mlflow_utils.py`). |

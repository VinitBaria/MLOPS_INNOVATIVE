# How to run locally (verified vs. not run)

## What was actually checked in this environment

- Repository checkout is clean before documentation work; the `data/movies.db` file has the five application tables (`movies`, `users`, `ratings`, `recommendation_logs`, `drift_logs`). It was inspected read-only.
- `python` is not on PATH. `python3` is Python **3.14.6**; the README says Python 3.11+ and the CI/Docker use 3.11.
- None of the imported project dependencies or `pytest` were installed in the available Python 3.14 environment.
- `python3 -m pytest tests/ -v` failed immediately with `/opt/homebrew/opt/python@3.14/bin/python3.14: No module named pytest`.
- `docker compose config --quiet` succeeded, with warning that `docker-compose.yml`'s top-level `version` attribute is obsolete and ignored.
- No tests, dependency installation, DB seed, MovieLens download, model training, Docker build, container startup, or deployment was run. Training/seed involve network or filesystem changes; Docker/Kubernetes actions start services or require external infrastructure. I did not modify existing code or data.

## Prerequisites and isolated local setup

Use Python 3.11 to match CI/container. `requirements.txt` has minimum versions, not a lock file. The commands below assume `python3.11`, `pip`, and network access to the package index and MovieLens download URL.

```bash
cd /Users/jenilbuha/Desktop/sem7/Mlops/assignment/MLOPS_INNOVATIVE
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To avoid modifying the tracked `data/movies.db`, set an isolated SQLite database path for commands that import the API/database package:

```bash
export DATABASE_URL=sqlite:////tmp/cinematch-local.db
```

`src/database/db.py` uses `DATABASE_URL` when present. Running without this override may attempt PostgreSQL then fall back to the repository's `data/movies.db`. Do not point setup/test commands at production data.

## Data setup

`run_project.py setup` downloads MovieLens if needed, creates the DB tables, then seeds movie/user rows and up to 15,000 baseline ratings. It writes to the configured database and `data/raw/`, and requires network access if data is absent (`run_project.py::main`, `src/database/init_db.py::init_database`, `src/data/download.py::download_movielens`). Use the isolated DB above:

```bash
python run_project.py setup
```

Expected code-level output includes messages for table creation, MovieLens download/extraction or existing dataset, and inserted movies/users/ratings. Actual output is **not verified** because required Python packages are missing and this would download data/write DB files.

The downloader uses `requests.get` with a 120-second timeout, checks HTTP status, extracts the ZIP, then deletes the ZIP (`src/data/download.py::download_movielens`). Review the external data source and available disk space before running; the command was not run here.

## Train and inspect the model

```bash
python src/training/train.py --no-mlflow
```

This downloads/reuses MovieLens, validates it, preprocesses it, trains all three recommenders, evaluates, selects by NDCG@10, and writes `models/artifacts/` and `data/processed/preprocessor.joblib` (`src/training/train.py::train_pipeline`). It can take materially longer and writes ignored model artifacts; it does not use the SQL feedback data. It was not run here, so duration, metrics, and exact output cannot be guaranteed. If you want MLflow tracking, start/configure a tracking server first and omit `--no-mlflow`; `run_project.py train` also disables MLflow by default.

## Serve the API

After successful training, keep `DATABASE_URL` pointed at the local DB and run:

```bash
python run_project.py serve
```

This starts Uvicorn on `0.0.0.0:8000` (`run_project.py::main`). The API requires DB tables/catalog rows and compatible model/preprocessor artifacts; with missing artifacts startup prints a notice and returns fallback recommendations. Visit `/docs`, `/health`, `/metrics`, `/movies`, and `/recommend/1?limit=5`. No server was started here.

The direct equivalent is:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Test

With dependencies installed and the isolated DB configured:

```bash
DATABASE_URL=sqlite:////tmp/cinematch-tests.db python -m pytest tests/ -v
```

The API test module imports the app, which calls `Base.metadata.create_all`; this explicit temporary database prevents those schema writes from changing the tracked DB. Tests may create/update a test rating in their DB. This command is a recommendation based on `tests/test_api.py` and `src/api/main.py`; it has not been run. The only attempted test command failed because pytest was unavailable.

## Docker Compose

Configuration-only validation:

```bash
docker compose config --quiet
```

This succeeded in the current environment with the obsolete `version` warning noted above. Starting the stack requires Docker daemon and may start containers, create volumes, bind ports, and write DB state; it was not run.

```bash
docker compose up --build
```

**Likely build blocker from repository inventory:** `Dockerfile` unconditionally copies `frontend/`, which is absent from this checkout. Also, API model artifacts are not included in the image; Compose expects `./models` mounted. Resolve/check those constraints before attempting a local build or run. This documentation task did not alter them.

## Not recommended without explicit review

- `python run_project.py all` downloads/seeds data, trains and writes model artifacts, then runs a public-interface server until stopped. It does not run tests.
- `python src/training/retrain.py` uses `force=True` under `__main__`; the retraining function can overwrite the production artifact directory before passing/failing its quality gate.
- Jenkins deployment and the README's `kubectl apply` sequence require a Kubernetes cluster and deployment credentials/access; they can change external infrastructure.
- MLflow, Postgres, Docker, and Kubernetes service startup may use ports/storage and require configuration.

No cloud service, paid API, credential, or cluster was used in this verification.

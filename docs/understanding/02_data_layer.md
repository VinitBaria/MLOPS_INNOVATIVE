# Data layer

## Source and schema

The only programmatic training source is MovieLens 100K from `https://files.grouplens.org/datasets/movielens/ml-100k.zip` (`src/data/download.py::download_movielens`). It is extracted under `data/raw/ml-100k`; download skips if that directory already exists, with no checksum or version verification.

| Source file / DB entity | Fields used | Role |
|---|---|---|
| MovieLens `u.data` | `userId`, `movieId`, integer `rating`, epoch `timestamp` (converted to datetime) | Ratings for training and evaluation (`src/data/download.py::load_ratings`). |
| MovieLens `u.item` | `movieId`, `title`, `release_date`, 19 binary genre flags; `video_release_date` and URL are dropped; combined pipe-separated `genres` is derived | Movie features (`src/data/download.py::load_movies`). |
| MovieLens `u.user` | `userId`, age, gender, occupation, `zip_code` | Demographics; loaded and used to seed DB, but not used in model feature engineering (`load_users`, `src/database/init_db.py::init_database`). |
| `movies` SQL table | internal PK, unique `movie_id`, title, optional release date, genre string | Catalog served by API (`src/database/models.py::Movie`). |
| `users` SQL table | internal PK, unique `user_id`, demographic columns | Seeded user metadata; rating FK target (`src/database/models.py::User`). |
| `ratings` SQL table | user/movie FKs, rating, `created_at`, `is_new` | Baseline and new user feedback (`src/database/models.py::Rating`). |
| `recommendation_logs`, `drift_logs` | recommendation IDs/model/latency/time; metric name/score/p-value/flag/time | Logs/data schema; `DriftLog` has no observed write call in the checked-in source. |

`src/database/db.py` tries PostgreSQL using `DATABASE_URL` or `POSTGRES_*`, unless `USE_SQLITE=true`; any PostgreSQL connection exception falls back to `data/movies.db`. The checked-in DB contains all five application tables, confirmed with `sqlite3 -readonly`; contents and row counts were not examined. `init_database(seed_data=True, limit_ratings=...)` seeds all movies/users but takes only the first N rows of ratings (`src/database/init_db.py::init_database`). `run_project.py setup` chooses N=15,000; the standalone initializer default is 10,000.

## Validation, cleaning, features

1. `src/training/train.py::train_pipeline` downloads/loads ratings and movies, but not demographic users.
2. `src/data/validate.py::DataValidator.validate_ratings` checks required columns, nulls, ratings in [1,5], duplicate user/movie pairs, minimums (10 users, 10 movies, 100 ratings), and rating concentration. Nulls/duplicates/concentrated ratings are warnings, but overall status is computed as all checks exactly `PASS`; any warning therefore makes ratings status `FAIL`. The training pipeline stops only when that report status is `FAIL`. These two rules imply warned data halts training, despite the warning labels.
3. `validate_movies` checks required movie ID/title, nulls, and unique IDs. It reports warnings for nulls, but the training code does not stop on a failed movie report; it only branches on `ratings_report`.
4. `DataPreprocessor._clean_ratings` sorts by timestamp, keeps the last duplicate user/movie record, drops ratings outside [1,5], then drops null user/movie/rating values.
5. `LabelEncoder` maps user and movie IDs to contiguous indices; encoder classes and dimensions are saved in `preprocessor.joblib`.
6. `fit_transform` defaults to `temporal_split=True`: one global timestamp sort and chronological 80/20 partition. Random split is available with `train_test_split` and seed 42 when temporal mode is disabled. Both encoders are fitted before the split, so test-only entities are known to the model mapping even if absent from training.
7. Train and test sparse user-item matrices are built from ratings; genre columns are the 19 MovieLens flags available in the movie frame, aligned by encoded movie ID. User/item demographic columns, text, posters, and timestamp-derived features are not model features.
8. Per-user train/test sets are generated for ranking evaluation. The model uses the train interaction matrix; test data supplies metrics.

**Why these design choices:** chronological partitioning has an evident purpose in the docstring and code (`_temporal_split`: “prevent data leakage”): it mimics predicting later events from earlier ratings. **Confirmed from repo (comments/code).** A global chronological holdout can leave some users with no training history; a per-user temporal split is an alternative with different evaluation coverage. **my inference.** ID encoding makes the matrix compact and row/column-indexable. **my inference** based on `LabelEncoder` and matrix shape; alternatives include sparse raw-ID maps or feature stores.

## Data/feature versioning

There is no DVC config, feature store, data manifest, checksum, or dataset snapshot. The raw ZIP and processed directory are ignored by `.gitignore`; the preprocessor is saved via joblib to `data/processed/preprocessor.joblib`, and training stores metrics/parameters/timestamp in `models/artifacts/metadata.json` (`src/data/preprocess.py::save`, `src/training/train.py::train_pipeline`). The dataset is not versioned alongside that metadata, so the exact training input cannot reliably be recreated from the model metadata alone.

**Why:** **confirmed from repo (code)** persistence of encoder state is needed to map request IDs consistently during serving; no repo source explains choosing joblib or omitting dataset versioning. Alternatives include DVC or immutable object-store snapshots, which improve reproducibility but require storage and pipeline configuration. **my inference.**

## Split and evaluation caveats

The declared default is 20% chronological holdout (`train_pipeline` and `DataPreprocessor.fit_transform`). No validation split or cross-validation exists. `DataPreprocessor` fits ID encoders on all cleaned records before splitting. `ModelEvaluator` uses every test movie as relevant regardless of its rating and does not use its `relevance_threshold` field; this means ranking metrics currently evaluate held-out interaction retrieval, not high-rating relevance. These are code observations, not just general recommendations (`src/models/evaluate.py::ModelEvaluator`).

The application DB ratings and the training file ratings are separate paths: new `POST /ratings` values are written to SQL, but `train_pipeline` reloads the MovieLens source and never queries the ratings table. Consequently, the current “new ratings trigger” does not by itself retrain on the feedback that triggered it (`src/api/main.py::submit_rating`, `src/training/train.py::train_pipeline`).

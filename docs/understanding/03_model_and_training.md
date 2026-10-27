# Models, training, and evaluation

## Approaches

The project fits three model classes (`src/training/train.py::train_pipeline`):

1. **Collaborative filtering (CF)** — `src/models/collaborative.py::CollaborativeFilteringModel` factorizes the user/movie rating matrix with truncated SVD. In intuition, users and movies that share rating patterns receive related latent vectors, so a user's scores can be estimated for movies they have not rated. This is interaction-driven; a new user/item has little or no learned signal.
2. **Content-based filtering (CB)** — `src/models/content_based.py::ContentBasedModel` uses binary movie genre vectors. It forms each user's genre profile as a rating-weighted average of genres of rated movies, then ranks genres by cosine similarity. This uses item metadata and can provide a signal for a user with limited ratings, but here it has no richer content such as plot text or actors.
3. **Hybrid recommender** — `src/models/hybrid.py::HybridRecommender` trains both models and mixes CF/CB candidate scores. Configured defaults are 0.7 CF / 0.3 content, with a five-rating cold-start threshold. For a user with fewer ratings, it linearly ramps the CF weight up; for an unknown encoded index it uses content only.

**Why these choices:** The comments/docstrings explain SVD's latent representations, genre similarity, and hybrid cold-start/adaptive-weight strategy. **Confirmed from repo (comments/code).** Alternatives include implicit-feedback ALS, nearest-neighbor CF, popularity baselines, learned ranking models, text/image embeddings, or a two-tower retrieval/ranking design. **my inference** about alternatives; none are implemented here. More complex methods may improve personalization or scale retrieval but require more data, features, infrastructure, and evaluation discipline.

## Architecture, parameters, scoring, and objective

### CF

`CollaborativeFilteringModel.fit` densifies the sparse interaction matrix, computes each user's mean over nonzero ratings, mean-centers only observed ratings, then runs `scipy.sparse.linalg.svds` with `k=min(n_factors, min(matrix_shape)-1)`. It reconstructs `U * diag(sigma) * Vt + user_mean` and clips all predictions to [1,5]. The CLI training default is `n_factors=50`; `run_project.py all` overrides it to 20. `svds` is a truncated matrix decomposition; the code contains no explicit optimizer, configured training loss, regularization term, or gradient-descent loop. Do not describe it as training with a specified RMSE/MSE loss.

Technical caveats: missing entries are zeros and remain zeros during decomposition; centering is only applied to observed entries. The full matrix is materialized dense despite sparse input. Thus the approach is simple for small data, but memory use grows with users × movies (`src/models/collaborative.py::fit`).

### Content-based and hybrid

`ContentBasedModel.fit` L2-normalizes genre rows to compute a movie/movie similarity matrix and constructs weighted user profiles using raw ratings. `predict` converts profile-to-movie cosine similarity to a nominal 1–5 score via `1 + 4 * max(0, similarity)`. There is no separately optimized content loss. `HybridRecommender.predict` linearly combines predicted rating-like scores.

`HybridRecommender.recommend` asks each component for up to `min(3*n, n_movies)` items and adds weighted scores only for movies returned by each component; the other component contributes zero if that movie is absent from its candidate list. Its final ordering is a sum of those candidate values. Configured weights do not get normalized if custom values fail to sum to one (`src/models/hybrid.py::recommend`, `_get_weights`). The model class defaults to 50 factors, weights 0.7/0.3, and threshold 5. The `train_pipeline` signature defaults to 50/0.7/0.3, while `run_project.py` sets factor count to 20.

**Why:** genre cosine similarity and weighted ensemble behavior are explicit code/docstring choices. **Confirmed from repo (comments/code).** Normalizing scores or using a learned ranking combiner would be alternatives; normalization may make component contributions more comparable, while a learned combiner needs a suitable validation protocol. **my inference.**

## Training flow

```mermaid
flowchart TD
  A[download_movielens] --> B[load ratings and movies]
  B --> C[DataValidator reports]
  C -->|rating status FAIL| X[Return failed]
  C -->|continue| D[Clean + encode IDs]
  D --> E[Global temporal train/test split]
  E --> F[CSR interaction matrix + genre vectors]
  F --> G[Fit CF]
  F --> H[Fit content model]
  F --> I[Fit hybrid (fits CF and content again)]
  G --> J[Evaluate all three]
  H --> J
  I --> J
  J --> K[Choose max NDCG@10]
  K --> L[Save all models, selected production copy, metadata]
  L --> M{use_mlflow?}
  M -->|yes| N[Log runs; register best run]
  M -->|no| O[Skip tracking]
```

Steps are implemented in `src/training/train.py::train_pipeline`; transforms and split are in `src/data/preprocess.py::DataPreprocessor.fit_transform`. The hybrid is independently refit after CF and content have already each been fit, so those calculations are repeated. Artifacts are written beneath `models/artifacts` by default; the preprocessor is saved separately under `data/processed`. The script creates no standalone validation set.

## Metrics and their actual use

`src/models/evaluate.py::ModelEvaluator.evaluate_all` computes metrics for K=5,10,20 by default when called from training (`train_pipeline`).

| Metric | Definition / interpretation | Implementation and use |
|---|---|---|
| RMSE | Square root of average squared prediction error; penalizes larger rating errors. | `ModelEvaluator._compute_rating_errors` compares `predict` with test ratings. |
| MAE | Average absolute rating error; less sensitive to large errors than RMSE. | Same method. |
| Precision@K | Relevant recommended items divided by K. | `_compute_ranking_metrics`; note denominator is K even if fewer than K items returned. |
| Recall@K | Relevant recommended items found divided by all held-out relevant items. | `_compute_ranking_metrics`. |
| NDCG@K | Discounted gain for relevant results, normalized by ideal ranking; higher rank contributes more. | `_ndcg_at_k`; relevance is binary membership in test set. |
| Coverage | Unique recommended catalog items divided by model catalog count. | `_compute_coverage`; samples at most 200 users with fixed random seed. |

`relevance_threshold=3.5` is accepted by `ModelEvaluator.__init__` but is never used. Therefore all test interactions count as relevant, regardless of rating. Training selects a winner solely by maximum NDCG@10, not a combination of metrics. The `ModelEvaluator._compute_rating_errors`, `_compute_ranking_metrics`, and `_compute_coverage` methods catch broad exceptions and skip failures; this may silently make metrics optimistic or empty. If no rating predictions complete, RMSE/MAE become infinity.

**Why:** NDCG is used as selection metric because `train_pipeline` explicitly names it the primary metric; the code does not document why it outranks RMSE, recall, or coverage. **Confirmed from repo (code) for metric choice; my inference for ranking-quality rationale.** Alternatives include rating error alone or business/offline metrics with graded relevance. Current holdout relevance makes ranking metrics less meaningful than the README's “quality gates” imply.

## Experiment tracking and registry

With `use_mlflow=True`, `src/training/train.py::train_pipeline` calls `setup_mlflow`, logs one MLflow run for each model through `src/training/mlflow_utils.py::log_training_run`, and attempts `register_model` on the best model under the name `hybrid-recommender`. A tracking URI comes from `MLFLOW_TRACKING_URI` or defaults to `sqlite:///mlflow.db`. Tracking errors are caught and local artifacts remain.

`register_model` creates a registered model version. `promote_model` has a helper to transition a version to `Production`, and `compare_models` computes a metric improvement, but neither is called by `train_pipeline` or `run_automated_retraining`. The saved local `production/` directory, not MLflow registry resolution, is loaded by `src.api.main::load_model_and_preprocessor`. MLflow staging-to-production deployment is therefore not implemented end to end.

## Cold-start handling

- Unknown user index in `HybridRecommender._get_weights` yields content-only recommendation; `ContentBasedModel.recommend` also falls back to movies with the most genre flags for empty or out-of-range user profiles.
- CF unknown users use catalog-average reconstructed scores (`CollaborativeFilteringModel._cold_start_recommend`).
- Unknown items in `predict` return a fallback (CF global mean / content score 0), but request mapping in the API only returns known movie encoder indices.
- No user demographic or explicit onboarding-preference cold start exists. The web/API fallback selects arbitrary DB rows rather than popularity (`src/api/main.py::get_recommendations`).

The implemented fallbacks are explicit. **Why:** the model comments identify cold start as the reason for content/popularity-like fallback. **Confirmed from repo (comments/code).** A true popularity baseline could be a simpler alternative for anonymous users; the API's current fallback is not actually popularity-based. **my inference.**

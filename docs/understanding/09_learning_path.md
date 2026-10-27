# Recommended learning path and interview questions

## Source reading order

Read a small vertical slice first, then follow the training lifecycle:

1. `README.md` — advertised intent and entry commands; verify every claim against implementation.
2. `run_project.py::main` — understand the supported actions and their order.
3. `requirements.txt`, `.env.example`, `.gitignore` — see dependencies, runtime knobs, and generated artifacts.
4. `src/database/models.py` and `src/database/db.py` — understand persistent entities and connection selection.
5. `src/data/download.py::load_ratings`, `load_movies`, `load_users` — inspect MovieLens input schemas.
6. `src/data/validate.py::DataValidator` — learn checks and status semantics.
7. `src/data/preprocess.py::DataPreprocessor.fit_transform` — follow cleaning, encoding, split, matrices, features.
8. `src/models/collaborative.py::CollaborativeFilteringModel.fit` — inspect the interaction factorization.
9. `src/models/content_based.py::ContentBasedModel.fit` — inspect genre user profiles and cosine ranking.
10. `src/models/hybrid.py::HybridRecommender.fit`, `_get_weights`, `recommend` — understand ensemble and cold start.
11. `src/models/evaluate.py::ModelEvaluator.evaluate_all` — identify what metrics truly measure.
12. `src/training/train.py::train_pipeline` — connect input to selected and saved artifacts.
13. `src/training/mlflow_utils.py` — distinguish logging, registration, stage promotion helpers.
14. `src/api/schemas.py`, then `src/api/main.py::load_model_and_preprocessor`, `get_recommendations`, and each route — trace online ID mapping.
15. `src/monitoring/metrics.py` and `src/monitoring/drift.py` — separate wired API metrics from standalone drift logic.
16. `src/training/retrain.py::run_automated_retraining` — review threshold checks and actual file mutation order.
17. `tests/test_models.py`, `test_data.py`, `test_drift.py`, `test_api.py` — see present test scope and omissions.
18. `docker-compose.yml`, `Dockerfile`, `docker/*`, `k8s/*`, `.github/workflows/ci-cd.yaml`, `Jenkinsfile` — compare deployment intent with runtime artifact expectations.

There is one commit only (`9c8aae8`), so commit history does not provide additional decision context.

## 25 interview-style questions and answers

1. **What does the system recommend from?**  
   Ratings and movie genres from MovieLens 100K (`src/data/download.py::load_ratings`, `load_movies`); demographics are stored but not used in model features.

2. **What is collaborative filtering here?**  
   A user/movie rating matrix is decomposed with truncated SVD; reconstruction supplies predicted scores (`src/models/collaborative.py::CollaborativeFilteringModel.fit`).

3. **What is the content signal?**  
   Binary genre flags; user genre profiles are rating-weighted averages and items are ranked with cosine similarity (`src/models/content_based.py::ContentBasedModel.fit`).

4. **How does hybrid cold start work?**  
   It ramps collaborative weight according to rating count below five and uses configured weights above that; unknown users get content-only weights (`src/models/hybrid.py::HybridRecommender._get_weights`).

5. **What is a latent factor?**  
   A learned/decomposed compact coordinate for user/item rating patterns; in this repo it is an SVD component (`src/models/collaborative.py`).

6. **Is this model trained with RMSE loss?**  
   No explicit loss optimizer is coded. SVD is computed by SciPy; RMSE is an evaluation metric (`src/models/collaborative.py::fit`, `src/models/evaluate.py::_compute_rating_errors`).

7. **How is the train/test split made?**  
   Global ascending timestamp order, with the latest 20% as test by default (`src/data/preprocess.py::DataPreprocessor._temporal_split`).

8. **Why use a temporal split?**  
   The function says it prevents data leakage (`_temporal_split`); it mimics predicting later records from earlier ones. That rationale is explicit, although global split limitations remain.

9. **What selects the “best” model?**  
   Maximum NDCG@10 across CF, content, and hybrid (`src/training/train.py::train_pipeline`).

10. **What counts as relevant in ranking metrics?**  
    Every held-out movie interaction. `relevance_threshold` is stored but unused (`src/models/evaluate.py::ModelEvaluator`).

11. **What are Precision@K and Recall@K?**  
    Precision is relevant items among top K divided by K; recall is relevant items among top K divided by all held-out relevant items (`ModelEvaluator._compute_ranking_metrics`).

12. **What does NDCG add over precision?**  
    It discounts relevant items lower in the list and normalizes by ideal ordering (`ModelEvaluator._ndcg_at_k`).

13. **How are raw user/movie IDs used by the model?**  
    Encoders map them to contiguous integer indices; saved encoders map request IDs and recommendations back (`DataPreprocessor`, `src/api/main.py::get_recommendations`).

14. **Where does the catalog live?**  
    SQLAlchemy `Movie` rows in PostgreSQL or SQLite (`src/database/models.py::Movie`, `src/database/db.py`).

15. **Does model training use ratings submitted through the API?**  
    No. Retraining calls the same MovieLens-file `train_pipeline`; it does not read DB ratings (`src/training/retrain.py`, `src/training/train.py`).

16. **What artifacts are needed to serve?**  
    `data/processed/preprocessor.joblib`, `models/artifacts/production` hybrid files, and JSON metadata (`src/api/main.py::load_model_and_preprocessor`).

17. **Where are experiments logged?**  
    MLflow utility functions can log params, metrics, tags, and artifacts; `train_pipeline` enables them only when requested (`src/training/mlflow_utils.py`, `src/training/train.py`).

18. **Does registration automatically deploy to production?**  
    No. A registration helper exists, but the API loads local disk artifacts; `promote_model` is not called from retraining (`src/training/mlflow_utils.py`, `src/api/main.py`).

19. **How does an API rating trigger retraining?**  
    It sets `is_new=1`, counts pending DB ratings, and schedules a background task at the threshold (`src/api/main.py::submit_rating`).

20. **Is drift-based retraining active?**  
    No. The KS/PSI detector is standalone and no checked-in path invokes it (`src/monitoring/drift.py`).

21. **What is KS test used for here?**  
    A two-sample test for differences between baseline and recent rating distributions (`detect_rating_drift`).

22. **What does PSI represent in this code?**  
    A binned comparison of expected vs actual rating proportions (`calculate_psi`).

23. **What online metrics are exposed?**  
    Request counts/latency, inference time, recommendations served, rating count, model quality gauges, and drift gauges (`src/monitoring/metrics.py`).

24. **What is the biggest release-safety issue?**  
    Retraining saves into the production path before the quality gate and never restores rejected artifacts (`src/training/retrain.py::run_automated_retraining`, `src/training/train.py::train_pipeline`).

25. **What should be addressed before calling this production-ready?**  
    Verify feedback enters training, make artifact promotion atomic/rollback-capable, align served model type with saved type, wire drift and real quality checks, secure endpoints/secrets, add artifact/data versioning, and test actual deployment prerequisites (`src/training/*`, `src/api/main.py`, configs).

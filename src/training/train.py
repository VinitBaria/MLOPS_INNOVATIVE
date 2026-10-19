"""
Training Pipeline
End-to-end training: download → validate → preprocess → train → evaluate → log.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.data.download import download_movielens, load_ratings, load_movies
from src.data.preprocess import DataPreprocessor
from src.data.validate import DataValidator, print_validation_report
from src.models.collaborative import CollaborativeFilteringModel
from src.models.content_based import ContentBasedModel
from src.models.hybrid import HybridRecommender
from src.models.evaluate import ModelEvaluator, print_evaluation_report


def train_pipeline(
    n_factors: int = 50,
    cf_weight: float = 0.7,
    cb_weight: float = 0.3,
    test_size: float = 0.2,
    temporal_split: bool = True,
    use_mlflow: bool = True,
    model_save_dir: str = None,
) -> dict:
    """
    Complete training pipeline.
    
    Steps:
        1. Download/load MovieLens data
        2. Validate data quality
        3. Preprocess and split data
        4. Train all three models
        5. Evaluate all models
        6. Log to MLflow (if enabled)
        7. Save the best model
        
    Args:
        n_factors: SVD latent factors
        cf_weight: Collaborative filtering weight in hybrid
        cb_weight: Content-based weight in hybrid
        test_size: Test split fraction
        temporal_split: Use temporal split vs random
        use_mlflow: Whether to log to MLflow
        model_save_dir: Where to save model artifacts
        
    Returns:
        Dictionary with all results
    """
    start_time = time.time()

    if model_save_dir is None:
        model_save_dir = os.path.join(PROJECT_ROOT, "models", "artifacts")

    print("=" * 70)
    print("  🎬 Movie Recommendation System — Training Pipeline")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ─── Step 1: Download/Load Data ───
    print("\n📥 Step 1: Loading MovieLens dataset...")
    data_dir = download_movielens()
    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)

    # ─── Step 2: Validate Data ───
    print("\n🔍 Step 2: Validating data...")
    validator = DataValidator()
    ratings_report = validator.validate_ratings(ratings)
    movies_report = validator.validate_movies(movies)

    print_validation_report(ratings_report)
    print_validation_report(movies_report)

    if ratings_report["status"] == "FAIL":
        print("❌ Data validation failed! Check the report above.")
        return {"status": "failed", "reason": "data_validation_failed"}

    # ─── Step 3: Preprocess ───
    print("\n⚙️  Step 3: Preprocessing data...")
    preprocessor = DataPreprocessor()
    data = preprocessor.fit_transform(
        ratings, movies,
        test_size=test_size,
        temporal_split=temporal_split,
    )

    # Save preprocessor for serving
    preprocessor.save()

    # ─── Step 4: Train Models ───
    print("\n🧠 Step 4: Training models...")

    # Model A: Collaborative Filtering
    print("\n--- Model A: Collaborative Filtering ---")
    cf_model = CollaborativeFilteringModel(n_factors=n_factors)
    cf_model.fit(data["train_matrix"])

    # Model B: Content-Based
    print("\n--- Model B: Content-Based Filtering ---")
    cb_model = ContentBasedModel()
    cb_model.fit(data["train_matrix"], data["genre_features"])

    # Model C: Hybrid
    print("\n--- Model C: Hybrid Recommender ---")
    hybrid_model = HybridRecommender(
        n_factors=n_factors,
        cf_weight=cf_weight,
        cb_weight=cb_weight,
    )
    hybrid_model.fit(data["train_matrix"], data["genre_features"])

    # ─── Step 5: Evaluate ───
    print("\n📊 Step 5: Evaluating models...")
    evaluator = ModelEvaluator(relevance_threshold=3.5)

    models = {
        "collaborative_filtering": cf_model,
        "content_based": cb_model,
        "hybrid_recommender": hybrid_model,
    }

    all_metrics = {}
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        metrics = evaluator.evaluate_all(
            model,
            test_user_items=data["test_user_items"],
            train_user_items=data["train_user_items"],
            test_df=data["test_df"],
            k_values=[5, 10, 20],
        )
        all_metrics[name] = metrics
        print_evaluation_report(metrics, name)

    # ─── Step 6: Select Best Model ───
    print("\n🏆 Step 6: Selecting best model...")
    primary_metric = "ndcg_at_10"

    best_model_name = max(
        all_metrics,
        key=lambda name: all_metrics[name].get(primary_metric, 0),
    )
    best_metrics = all_metrics[best_model_name]
    best_model = models[best_model_name]

    print(f"  Best model: {best_model_name}")
    print(f"  {primary_metric}: {best_metrics.get(primary_metric, 0):.4f}")

    # ─── Step 7: Save Models ───
    print("\n💾 Step 7: Saving models...")

    for name, model in models.items():
        save_path = os.path.join(model_save_dir, name)
        model.save(save_path)

    # Save the best model as "production"
    production_path = os.path.join(model_save_dir, "production")
    best_model.save(production_path)
    print(f"  Production model saved to {production_path}")

    # Save metadata
    metadata = {
        "best_model": best_model_name,
        "primary_metric": primary_metric,
        "metrics": all_metrics,
        "params": {
            "n_factors": n_factors,
            "cf_weight": cf_weight,
            "cb_weight": cb_weight,
            "test_size": test_size,
            "temporal_split": temporal_split,
        },
        "timestamp": datetime.now().isoformat(),
        "n_users": data["n_users"],
        "n_movies": data["n_movies"],
        "n_train": len(data["train_df"]),
        "n_test": len(data["test_df"]),
    }

    metadata_path = os.path.join(model_save_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # ─── Step 8: MLflow Logging ───
    if use_mlflow:
        print("\n📋 Step 8: Logging to MLflow...")
        try:
            from src.training.mlflow_utils import (
                setup_mlflow, log_training_run, register_model
            )

            setup_mlflow()

            for name, model in models.items():
                model_metrics = all_metrics[name]
                model_params = model.get_params()

                tags = {"model_type": name}
                if name == best_model_name:
                    tags["best_model"] = "true"

                run_id = log_training_run(
                    model=model,
                    metrics=model_metrics,
                    params=model_params,
                    model_path=os.path.join(model_save_dir, name),
                    run_name=f"train_{name}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                    tags=tags,
                )

                # Register the best model
                if name == best_model_name:
                    try:
                        version = register_model(run_id, "hybrid-recommender")
                        metadata["mlflow_run_id"] = run_id
                        metadata["mlflow_model_version"] = version
                    except Exception as e:
                        print(f"  Warning: Model registration failed: {e}")

            print("  MLflow logging complete.")
        except Exception as e:
            print(f"  Warning: MLflow logging failed (server may not be running): {e}")
            print("  Models are still saved locally. Run MLflow server and retry.")
    else:
        print("\n⏭️  Step 8: MLflow logging skipped.")

    # ─── Summary ───
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("  ✅ Training Pipeline Complete!")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Best Model: {best_model_name}")
    print(f"  NDCG@10: {best_metrics.get('ndcg_at_10', 0):.4f}")
    print(f"  RMSE: {best_metrics.get('rmse', 0):.4f}")
    print(f"  Models saved to: {model_save_dir}")
    print("=" * 70)

    # Update metadata with timing
    metadata["elapsed_seconds"] = elapsed
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return {
        "status": "success",
        "best_model": best_model_name,
        "metrics": all_metrics,
        "metadata": metadata,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Movie Recommendation Models")
    parser.add_argument("--n-factors", type=int, default=50, help="SVD latent factors")
    parser.add_argument("--cf-weight", type=float, default=0.7, help="CF weight in hybrid")
    parser.add_argument("--cb-weight", type=float, default=0.3, help="CB weight in hybrid")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    parser.add_argument("--random-split", action="store_true", help="Use random split instead of temporal")

    args = parser.parse_args()

    result = train_pipeline(
        n_factors=args.n_factors,
        cf_weight=args.cf_weight,
        cb_weight=args.cb_weight,
        test_size=args.test_size,
        temporal_split=not args.random_split,
        use_mlflow=not args.no_mlflow,
    )

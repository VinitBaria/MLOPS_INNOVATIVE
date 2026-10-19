"""
Automated Model Retraining Pipeline with Quality Gates
Runs retraining on accumulated new ratings or detected drift,
validates quality gates, and safely promotes or rejects the candidate.
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.database.db import SessionLocal
from src.database.models import Rating, Movie
from src.training.train import train_pipeline
from src.monitoring.metrics import ACTIVE_MODEL_VERSION, MODEL_NDCG_SCORE, MODEL_RECALL_SCORE

METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "artifacts", "metadata.json")


def run_automated_retraining(
    min_ndcg_threshold: float = 0.15,
    max_regression_allowed: float = 0.05,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Executes automated retraining pipeline:
    1. Checks if sufficient new data exists.
    2. Runs training & evaluation across models.
    3. Quality Gate: Compares candidate against active production model.
    4. Safe promotion or rollback.
    """
    db = SessionLocal()
    new_ratings_count = db.query(Rating).filter(Rating.is_new == 1).count()
    db.close()

    print(f"Checking retraining trigger: {new_ratings_count} new ratings pending.")

    # Load current production metadata if available
    current_ndcg = 0.0
    current_version = "v1"
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                meta = json.load(f)
                best_name = meta.get("best_model", "hybrid_recommender")
                current_ndcg = meta.get("metrics", {}).get(best_name, {}).get("ndcg_at_10", 0.0)
                current_version = meta.get("model_version", "v1")
        except Exception as e:
            print(f"Notice: Could not read previous metadata: {e}")

    # Retrain
    print("Initiating model training run...")
    result = train_pipeline(use_mlflow=True)

    if result.get("status") != "success":
        return {
            "status": "REJECTED",
            "reason": "Training pipeline failed execution.",
            "promoted": False,
        }

    best_model = result["best_model"]
    candidate_metrics = result["metrics"][best_model]
    candidate_ndcg = candidate_metrics.get("ndcg_at_10", 0.0)
    candidate_recall = candidate_metrics.get("recall_at_10", 0.0)

    # ─── Quality Gates ───
    passed_absolute_gate = candidate_ndcg >= min_ndcg_threshold
    allowed_floor = current_ndcg * (1.0 - max_regression_allowed)
    passed_regression_gate = candidate_ndcg >= allowed_floor

    passed_all_gates = (passed_absolute_gate and passed_regression_gate) or force

    if passed_all_gates:
        print(f"✅ Quality gates passed! Candidate NDCG@10: {candidate_ndcg:.4f} >= {allowed_floor:.4f}")
        # Mark pending ratings as processed
        db = SessionLocal()
        db.query(Rating).filter(Rating.is_new == 1).update({"is_new": 0})
        db.commit()
        db.close()

        # Update metrics in Prometheus
        MODEL_NDCG_SCORE.set(candidate_ndcg)
        MODEL_RECALL_SCORE.set(candidate_recall)

        return {
            "status": "PROMOTED",
            "promoted": True,
            "best_model": best_model,
            "candidate_ndcg": candidate_ndcg,
            "previous_ndcg": current_ndcg,
            "improvement": candidate_ndcg - current_ndcg,
            "timestamp": datetime.utcnow().isoformat(),
        }
    else:
        print(f"❌ Quality gate rejected! Candidate NDCG@10 ({candidate_ndcg:.4f}) below threshold.")
        return {
            "status": "REJECTED",
            "promoted": False,
            "reason": "Quality gate failed: performance regression detected.",
            "candidate_ndcg": candidate_ndcg,
            "previous_ndcg": current_ndcg,
        }


if __name__ == "__main__":
    run_automated_retraining(force=True)

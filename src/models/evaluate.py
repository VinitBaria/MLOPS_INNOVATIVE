"""
Model Evaluation Module
Computes ranking metrics: RMSE, Precision@K, Recall@K, NDCG@K, Coverage.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict


class ModelEvaluator:
    """
    Evaluates recommendation models using both rating-prediction
    and ranking-based metrics.
    
    Metrics:
        - RMSE: Root Mean Squared Error on predicted ratings
        - MAE: Mean Absolute Error on predicted ratings
        - Precision@K: Fraction of recommended items that are relevant
        - Recall@K: Fraction of relevant items that are recommended
        - NDCG@K: Normalized Discounted Cumulative Gain
        - Coverage: Fraction of all items recommended to at least one user
        - Diversity: Average dissimilarity of recommended items
    """

    def __init__(self, relevance_threshold: float = 3.5):
        """
        Args:
            relevance_threshold: Minimum rating to consider an item 'relevant'
        """
        self.relevance_threshold = relevance_threshold

    def evaluate_all(
        self,
        model,
        test_user_items: Dict[int, Set[int]],
        train_user_items: Dict[int, Set[int]],
        test_df=None,
        k_values: List[int] = None,
    ) -> Dict[str, float]:
        """
        Run all evaluation metrics.
        
        Args:
            model: Trained recommendation model with .recommend() and .predict()
            test_user_items: {user_idx: set of movie indices in test}
            train_user_items: {user_idx: set of movie indices in train}
            test_df: Test DataFrame for RMSE computation (optional)
            k_values: List of K values for @K metrics
            
        Returns:
            Dictionary of metric_name -> value
        """
        if k_values is None:
            k_values = [5, 10, 20]

        metrics = {}

        # RMSE and MAE (if test_df provided)
        if test_df is not None:
            rmse, mae = self._compute_rating_errors(model, test_df)
            metrics["rmse"] = rmse
            metrics["mae"] = mae

        # Ranking metrics for each K
        for k in k_values:
            precision, recall, ndcg = self._compute_ranking_metrics(
                model, test_user_items, train_user_items, k
            )
            metrics[f"precision_at_{k}"] = precision
            metrics[f"recall_at_{k}"] = recall
            metrics[f"ndcg_at_{k}"] = ndcg

        # Coverage
        coverage = self._compute_coverage(
            model, train_user_items, k=max(k_values)
        )
        metrics["coverage"] = coverage

        return metrics

    def _compute_rating_errors(self, model, test_df) -> Tuple[float, float]:
        """Compute RMSE and MAE on test ratings."""
        actuals = []
        predictions = []

        for _, row in test_df.iterrows():
            user_idx = int(row["user_idx"])
            movie_idx = int(row["movie_idx"])
            actual = float(row["rating"])

            try:
                pred = model.predict(user_idx, movie_idx)
                actuals.append(actual)
                predictions.append(pred)
            except Exception:
                continue

        if not actuals:
            return float("inf"), float("inf")

        actuals = np.array(actuals)
        predictions = np.array(predictions)

        rmse = float(np.sqrt(np.mean((actuals - predictions) ** 2)))
        mae = float(np.mean(np.abs(actuals - predictions)))

        return rmse, mae

    def _compute_ranking_metrics(
        self,
        model,
        test_user_items: Dict[int, Set[int]],
        train_user_items: Dict[int, Set[int]],
        k: int,
    ) -> Tuple[float, float, float]:
        """
        Compute Precision@K, Recall@K, and NDCG@K averaged across users.
        """
        precisions = []
        recalls = []
        ndcgs = []

        for user_idx in test_user_items:
            # Only evaluate users with test items
            relevant = test_user_items[user_idx]
            if not relevant:
                continue

            # Get recommendations excluding training items
            watched = train_user_items.get(user_idx, set())

            try:
                recs = model.recommend(user_idx, n=k, exclude_watched=watched)
            except Exception:
                continue

            rec_items = [item_idx for item_idx, _ in recs]

            # Precision@K
            n_relevant_in_recs = sum(1 for item in rec_items if item in relevant)
            precision = n_relevant_in_recs / k if k > 0 else 0
            precisions.append(precision)

            # Recall@K
            recall = n_relevant_in_recs / len(relevant) if relevant else 0
            recalls.append(recall)

            # NDCG@K
            ndcg = self._ndcg_at_k(rec_items, relevant, k)
            ndcgs.append(ndcg)

        avg_precision = float(np.mean(precisions)) if precisions else 0.0
        avg_recall = float(np.mean(recalls)) if recalls else 0.0
        avg_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0

        return avg_precision, avg_recall, avg_ndcg

    def _ndcg_at_k(
        self, recommended: List[int], relevant: Set[int], k: int
    ) -> float:
        """Compute NDCG@K for a single user."""
        dcg = 0.0
        for i, item in enumerate(recommended[:k]):
            if item in relevant:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0

        # Ideal DCG
        ideal_relevant = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_relevant))

        return dcg / idcg if idcg > 0 else 0.0

    def _compute_coverage(
        self,
        model,
        train_user_items: Dict[int, Set[int]],
        k: int = 10,
        sample_users: int = 200,
    ) -> float:
        """
        Compute catalog coverage: fraction of items recommended
        to at least one user.
        """
        all_recommended = set()
        users = list(train_user_items.keys())

        # Sample users for efficiency
        if len(users) > sample_users:
            rng = np.random.RandomState(42)
            users = rng.choice(users, size=sample_users, replace=False)

        for user_idx in users:
            watched = train_user_items.get(user_idx, set())
            try:
                recs = model.recommend(user_idx, n=k, exclude_watched=watched)
                for item_idx, _ in recs:
                    all_recommended.add(item_idx)
            except Exception:
                continue

        total_items = model.n_movies if hasattr(model, "n_movies") else 1
        coverage = len(all_recommended) / total_items if total_items > 0 else 0.0

        return float(coverage)


def print_evaluation_report(metrics: Dict[str, float], model_name: str = "Model"):
    """Print a formatted evaluation report."""
    print(f"\n{'=' * 50}")
    print(f"  Evaluation Report: {model_name}")
    print(f"{'=' * 50}")

    # Rating prediction metrics
    if "rmse" in metrics:
        print(f"\n  Rating Prediction:")
        print(f"    RMSE:  {metrics['rmse']:.4f}")
        print(f"    MAE:   {metrics['mae']:.4f}")

    # Ranking metrics
    k_values = sorted(set(
        int(k.split("_")[-1])
        for k in metrics
        if k.startswith("precision_at_")
    ))

    if k_values:
        print(f"\n  Ranking Metrics:")
        print(f"    {'K':<6} {'Prec@K':<10} {'Recall@K':<10} {'NDCG@K':<10}")
        print(f"    {'-'*36}")
        for k in k_values:
            p = metrics.get(f"precision_at_{k}", 0)
            r = metrics.get(f"recall_at_{k}", 0)
            n = metrics.get(f"ndcg_at_{k}", 0)
            print(f"    {k:<6} {p:<10.4f} {r:<10.4f} {n:<10.4f}")

    if "coverage" in metrics:
        print(f"\n  Coverage: {metrics['coverage']:.4f}")

    print(f"{'=' * 50}\n")

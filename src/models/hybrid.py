"""
Hybrid Recommender
Combines Collaborative Filtering and Content-Based Filtering
with weighted scoring and cold-start fallback.
"""

import numpy as np
from scipy.sparse import csr_matrix
import joblib
import os
from typing import List, Tuple, Dict, Optional

from src.models.collaborative import CollaborativeFilteringModel
from src.models.content_based import ContentBasedModel


class HybridRecommender:
    """
    Hybrid Recommendation Model combining CF and CB approaches.
    
    Strategy:
        - For users with sufficient ratings: weighted combination of CF + CB scores
        - For cold-start users (few/no ratings): fall back to content-based
        - For cold-start items (no interactions): fall back to content-based
    
    Parameters:
        n_factors: Number of SVD latent factors for CF
        cf_weight: Weight for collaborative filtering scores (0-1)
        cb_weight: Weight for content-based scores (0-1)
        cold_start_threshold: Minimum ratings before using CF
    """

    def __init__(
        self,
        n_factors: int = 50,
        cf_weight: float = 0.7,
        cb_weight: float = 0.3,
        cold_start_threshold: int = 5,
    ):
        self.n_factors = n_factors
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.cold_start_threshold = cold_start_threshold

        self.cf_model = CollaborativeFilteringModel(n_factors=n_factors)
        self.cb_model = ContentBasedModel()

        self.user_rating_counts = None
        self.n_users = 0
        self.n_movies = 0
        self._fitted = False

    def fit(
        self,
        interaction_matrix: csr_matrix,
        genre_features: np.ndarray,
    ) -> "HybridRecommender":
        """
        Train both sub-models.
        
        Args:
            interaction_matrix: Sparse user-item rating matrix
            genre_features: Movie genre feature matrix
            
        Returns:
            self
        """
        print("=" * 60)
        print("Training Hybrid Recommender")
        print("=" * 60)

        self.n_users = interaction_matrix.shape[0]
        self.n_movies = interaction_matrix.shape[1]

        # Count ratings per user (for cold-start detection)
        self.user_rating_counts = np.array(
            (interaction_matrix > 0).sum(axis=1)
        ).flatten()

        # Train collaborative filtering
        self.cf_model.fit(interaction_matrix)

        # Train content-based filtering
        self.cb_model.fit(interaction_matrix, genre_features)

        self._fitted = True
        print("=" * 60)
        print(f"Hybrid model ready. CF weight={self.cf_weight}, CB weight={self.cb_weight}")
        print(f"Cold-start threshold: {self.cold_start_threshold} ratings")
        print("=" * 60)

        return self

    def predict(self, user_idx: int, movie_idx: int) -> float:
        """
        Predict rating for a user-movie pair using hybrid scoring.
        
        Uses adaptive weighting: more CF weight for active users,
        more CB weight for cold-start users.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        cf_weight, cb_weight = self._get_weights(user_idx)

        cf_score = self.cf_model.predict(user_idx, movie_idx)
        cb_score = self.cb_model.predict(user_idx, movie_idx)

        return cf_weight * cf_score + cb_weight * cb_score

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_watched: Optional[set] = None,
    ) -> List[Tuple[int, float]]:
        """
        Get top-N hybrid recommendations.
        
        Args:
            user_idx: User index
            n: Number of recommendations to return
            exclude_watched: Set of movie indices to exclude
            
        Returns:
            List of (movie_idx, predicted_score) tuples, sorted by score
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        cf_weight, cb_weight = self._get_weights(user_idx)

        # Get more candidates than needed from each model
        n_candidates = min(n * 3, self.n_movies)

        cf_recs = self.cf_model.recommend(user_idx, n_candidates, exclude_watched)
        cb_recs = self.cb_model.recommend(user_idx, n_candidates, exclude_watched)

        # Combine scores
        combined_scores = {}

        for movie_idx, score in cf_recs:
            combined_scores[movie_idx] = combined_scores.get(movie_idx, 0) + cf_weight * score

        for movie_idx, score in cb_recs:
            combined_scores[movie_idx] = combined_scores.get(movie_idx, 0) + cb_weight * score

        # Sort by combined score
        sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_recs[:n]

    def _get_weights(self, user_idx: int) -> Tuple[float, float]:
        """
        Get adaptive weights based on user activity.
        
        Cold-start users get more CB weight; active users get more CF weight.
        """
        if user_idx >= self.n_users:
            # Unknown user — fully content-based
            return 0.0, 1.0

        n_ratings = self.user_rating_counts[user_idx]

        if n_ratings < self.cold_start_threshold:
            # Cold start: interpolate between CB-only and full hybrid
            ratio = n_ratings / self.cold_start_threshold
            cf_w = self.cf_weight * ratio
            cb_w = 1.0 - cf_w
            return cf_w, cb_w

        # Active user: use configured weights
        return self.cf_weight, self.cb_weight

    def get_similar_movies(
        self, movie_idx: int, n: int = 10
    ) -> List[Tuple[int, float]]:
        """Find similar movies (delegates to content-based model)."""
        return self.cb_model.get_similar_movies(movie_idx, n)

    def get_params(self) -> Dict:
        """Return all model parameters for MLflow logging."""
        return {
            "model_type": "hybrid_recommender",
            "n_factors": self.n_factors,
            "cf_weight": self.cf_weight,
            "cb_weight": self.cb_weight,
            "cold_start_threshold": self.cold_start_threshold,
            "n_users": self.n_users,
            "n_movies": self.n_movies,
        }

    def save(self, path: str):
        """Save complete hybrid model."""
        os.makedirs(path, exist_ok=True)

        # Save sub-models
        self.cf_model.save(path)
        self.cb_model.save(path)

        # Save hybrid state
        state = {
            "n_factors": self.n_factors,
            "cf_weight": self.cf_weight,
            "cb_weight": self.cb_weight,
            "cold_start_threshold": self.cold_start_threshold,
            "user_rating_counts": self.user_rating_counts,
            "n_users": self.n_users,
            "n_movies": self.n_movies,
        }
        joblib.dump(state, os.path.join(path, "hybrid_model.joblib"))
        print(f"Hybrid model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "HybridRecommender":
        """Load complete hybrid model."""
        state = joblib.load(os.path.join(path, "hybrid_model.joblib"))

        model = cls(
            n_factors=state["n_factors"],
            cf_weight=state["cf_weight"],
            cb_weight=state["cb_weight"],
            cold_start_threshold=state["cold_start_threshold"],
        )

        model.cf_model = CollaborativeFilteringModel.load(path)
        model.cb_model = ContentBasedModel.load(path)
        model.user_rating_counts = state["user_rating_counts"]
        model.n_users = state["n_users"]
        model.n_movies = state["n_movies"]
        model._fitted = True

        return model

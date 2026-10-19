"""
Collaborative Filtering Recommender
Uses SVD matrix factorization to learn latent factors for users and items.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import joblib
import os
from typing import List, Tuple, Dict, Optional


class CollaborativeFilteringModel:
    """
    SVD-based Collaborative Filtering Recommender.
    
    Learns latent factor representations for users and items from the
    user-item interaction matrix. Predictions are made by reconstructing
    the interaction matrix from the latent factors.
    
    Parameters:
        n_factors: Number of latent factors (default: 50)
        
    How it works:
        1. Decompose user-item matrix R ≈ U × Σ × Vt
        2. U captures user preferences in latent space
        3. V captures item characteristics in latent space
        4. Predict ratings by: predicted = U × Σ × Vt
    """

    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self.user_factors = None  # U matrix
        self.sigma = None  # Σ diagonal
        self.item_factors = None  # Vt matrix
        self.user_means = None  # Per-user rating means
        self.global_mean = 0.0
        self.predicted_ratings = None
        self._fitted = False

    def fit(self, interaction_matrix: csr_matrix) -> "CollaborativeFilteringModel":
        """
        Train the collaborative filtering model.
        
        Args:
            interaction_matrix: Sparse user-item matrix (users x movies)
            
        Returns:
            self
        """
        print(f"Training Collaborative Filtering (SVD, k={self.n_factors})...")

        # Convert to dense for mean computation
        R = interaction_matrix.toarray().astype(float)

        # Compute per-user mean (only non-zero entries)
        user_ratings_count = (R > 0).sum(axis=1)
        user_ratings_sum = R.sum(axis=1)

        # Avoid division by zero
        self.user_means = np.where(
            user_ratings_count > 0,
            user_ratings_sum / user_ratings_count,
            0.0,
        )
        self.global_mean = R[R > 0].mean()

        # Mean-center the matrix (only non-zero entries)
        R_centered = R.copy()
        for i in range(R.shape[0]):
            mask = R[i, :] > 0
            R_centered[i, mask] -= self.user_means[i]

        # Ensure n_factors doesn't exceed matrix dimensions
        k = min(self.n_factors, min(R.shape) - 1)

        # SVD decomposition
        U, sigma, Vt = svds(csr_matrix(R_centered), k=k)

        # Store factors
        self.user_factors = U
        self.sigma = sigma
        self.item_factors = Vt

        # Reconstruct predicted ratings
        sigma_diag = np.diag(sigma)
        self.predicted_ratings = (
            np.dot(np.dot(U, sigma_diag), Vt) + self.user_means.reshape(-1, 1)
        )

        # Clip to valid rating range
        self.predicted_ratings = np.clip(self.predicted_ratings, 1, 5)

        self._fitted = True
        print(f"  Training complete. Matrix shape: {R.shape}, Factors: {k}")

        return self

    def predict(self, user_idx: int, movie_idx: int) -> float:
        """Predict rating for a user-movie pair."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if user_idx >= self.predicted_ratings.shape[0]:
            return self.global_mean
        if movie_idx >= self.predicted_ratings.shape[1]:
            return self.global_mean

        return float(self.predicted_ratings[user_idx, movie_idx])

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_watched: Optional[set] = None,
    ) -> List[Tuple[int, float]]:
        """
        Get top-N recommendations for a user.
        
        Args:
            user_idx: User index
            n: Number of recommendations
            exclude_watched: Set of movie indices to exclude
            
        Returns:
            List of (movie_idx, predicted_rating) tuples
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if user_idx >= self.predicted_ratings.shape[0]:
            # Cold start: return highest global ratings
            return self._cold_start_recommend(n, exclude_watched)

        scores = self.predicted_ratings[user_idx].copy()

        # Exclude already watched movies
        if exclude_watched:
            for idx in exclude_watched:
                if idx < len(scores):
                    scores[idx] = -np.inf

        # Get top-N indices
        top_indices = np.argsort(scores)[::-1][:n]

        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def _cold_start_recommend(
        self, n: int, exclude: Optional[set] = None
    ) -> List[Tuple[int, float]]:
        """Recommend popular items for cold-start users."""
        avg_ratings = self.predicted_ratings.mean(axis=0)

        if exclude:
            for idx in exclude:
                if idx < len(avg_ratings):
                    avg_ratings[idx] = -np.inf

        top_indices = np.argsort(avg_ratings)[::-1][:n]
        return [(int(idx), float(avg_ratings[idx])) for idx in top_indices]

    def get_params(self) -> Dict:
        """Return model parameters for MLflow logging."""
        return {
            "model_type": "collaborative_filtering_svd",
            "n_factors": self.n_factors,
            "n_users": self.predicted_ratings.shape[0] if self._fitted else 0,
            "n_movies": self.predicted_ratings.shape[1] if self._fitted else 0,
        }

    def save(self, path: str):
        """Save model to disk."""
        os.makedirs(path, exist_ok=True)
        state = {
            "n_factors": self.n_factors,
            "user_factors": self.user_factors,
            "sigma": self.sigma,
            "item_factors": self.item_factors,
            "user_means": self.user_means,
            "global_mean": self.global_mean,
            "predicted_ratings": self.predicted_ratings,
        }
        joblib.dump(state, os.path.join(path, "collaborative_model.joblib"))
        print(f"Collaborative model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "CollaborativeFilteringModel":
        """Load model from disk."""
        state = joblib.load(os.path.join(path, "collaborative_model.joblib"))
        model = cls(n_factors=state["n_factors"])
        model.user_factors = state["user_factors"]
        model.sigma = state["sigma"]
        model.item_factors = state["item_factors"]
        model.user_means = state["user_means"]
        model.global_mean = state["global_mean"]
        model.predicted_ratings = state["predicted_ratings"]
        model._fitted = True
        return model

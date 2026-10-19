"""
Content-Based Filtering Recommender
Recommends movies based on genre similarity using cosine similarity.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import joblib
import os
from typing import List, Tuple, Dict, Optional


class ContentBasedModel:
    """
    Content-Based Filtering using movie genre features.
    
    Builds a user profile from their rated movies' genres, then
    recommends movies with similar genre profiles.
    
    How it works:
        1. For each user, compute a weighted average of genre vectors
           of their rated movies (weighted by rating).
        2. Compute cosine similarity between the user profile and
           all movie genre vectors.
        3. Rank movies by similarity score.
    """

    def __init__(self):
        self.genre_features = None  # Movie genre matrix (n_movies x n_genres)
        self.movie_similarity = None  # Movie-movie similarity matrix
        self.user_profiles = None  # User genre preference profiles
        self.n_users = 0
        self.n_movies = 0
        self._fitted = False

    def fit(
        self,
        interaction_matrix: csr_matrix,
        genre_features: np.ndarray,
    ) -> "ContentBasedModel":
        """
        Train the content-based model.
        
        Args:
            interaction_matrix: Sparse user-item rating matrix
            genre_features: Movie genre matrix (n_movies x n_genres)
            
        Returns:
            self
        """
        print("Training Content-Based Filter...")

        self.genre_features = genre_features
        self.n_users = interaction_matrix.shape[0]
        self.n_movies = interaction_matrix.shape[1]

        # Compute movie-movie similarity based on genres
        # Add small epsilon to avoid zero vectors
        genre_norm = genre_features.copy()
        row_norms = np.linalg.norm(genre_norm, axis=1, keepdims=True)
        row_norms = np.where(row_norms == 0, 1, row_norms)
        genre_norm = genre_norm / row_norms

        self.movie_similarity = cosine_similarity(genre_norm)

        # Build user profiles from their ratings
        R = interaction_matrix.toarray().astype(float)
        self.user_profiles = np.zeros((self.n_users, genre_features.shape[1]))

        for user_idx in range(self.n_users):
            rated_mask = R[user_idx, :] > 0
            if rated_mask.sum() == 0:
                continue

            # Weighted average of genre vectors (weighted by rating)
            ratings = R[user_idx, rated_mask]
            genres = genre_features[rated_mask]

            self.user_profiles[user_idx] = np.average(genres, axis=0, weights=ratings)

        self._fitted = True
        print(f"  Training complete. Built {self.n_users} user profiles, "
              f"Genre dimensions: {genre_features.shape[1]}")

        return self

    def predict(self, user_idx: int, movie_idx: int) -> float:
        """
        Predict relevance score for a user-movie pair.
        
        Returns a similarity score between 0 and 1.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if user_idx >= self.n_users or movie_idx >= self.n_movies:
            return 0.0

        user_profile = self.user_profiles[user_idx].reshape(1, -1)
        movie_genre = self.genre_features[movie_idx].reshape(1, -1)

        # Handle zero vectors
        if np.linalg.norm(user_profile) == 0 or np.linalg.norm(movie_genre) == 0:
            return 0.0

        similarity = cosine_similarity(user_profile, movie_genre)[0, 0]

        # Scale to 1-5 rating range
        return float(1 + 4 * max(0, similarity))

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_watched: Optional[set] = None,
    ) -> List[Tuple[int, float]]:
        """
        Get top-N content-based recommendations for a user.
        
        Args:
            user_idx: User index
            n: Number of recommendations
            exclude_watched: Set of movie indices to exclude
            
        Returns:
            List of (movie_idx, score) tuples
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if user_idx >= self.n_users:
            return self._cold_start_recommend(n, exclude_watched)

        user_profile = self.user_profiles[user_idx].reshape(1, -1)

        # Handle zero profile (no ratings)
        if np.linalg.norm(user_profile) == 0:
            return self._cold_start_recommend(n, exclude_watched)

        # Compute similarity with all movies
        scores = cosine_similarity(user_profile, self.genre_features)[0]

        # Exclude watched movies
        if exclude_watched:
            for idx in exclude_watched:
                if idx < len(scores):
                    scores[idx] = -np.inf

        # Get top-N
        top_indices = np.argsort(scores)[::-1][:n]

        # Scale scores to rating range
        return [
            (int(idx), float(1 + 4 * max(0, scores[idx])))
            for idx in top_indices
            if scores[idx] > -np.inf
        ]

    def get_similar_movies(
        self, movie_idx: int, n: int = 10
    ) -> List[Tuple[int, float]]:
        """Find movies similar to a given movie based on genre."""
        if not self._fitted or self.movie_similarity is None:
            raise RuntimeError("Model not fitted.")

        if movie_idx >= self.n_movies:
            return []

        similarities = self.movie_similarity[movie_idx].copy()
        similarities[movie_idx] = -1  # Exclude self

        top_indices = np.argsort(similarities)[::-1][:n]

        return [
            (int(idx), float(similarities[idx]))
            for idx in top_indices
        ]

    def _cold_start_recommend(
        self, n: int, exclude: Optional[set] = None
    ) -> List[Tuple[int, float]]:
        """Recommend diverse popular genres for cold-start users."""
        # Pick movies that cover the most genres
        genre_diversity = self.genre_features.sum(axis=1)

        if exclude:
            for idx in exclude:
                if idx < len(genre_diversity):
                    genre_diversity[idx] = -np.inf

        top_indices = np.argsort(genre_diversity)[::-1][:n]
        return [(int(idx), 3.0) for idx in top_indices]

    def get_params(self) -> Dict:
        """Return model parameters for MLflow logging."""
        return {
            "model_type": "content_based_cosine",
            "n_genres": self.genre_features.shape[1] if self._fitted else 0,
            "n_users": self.n_users,
            "n_movies": self.n_movies,
        }

    def save(self, path: str):
        """Save model to disk."""
        os.makedirs(path, exist_ok=True)
        state = {
            "genre_features": self.genre_features,
            "movie_similarity": self.movie_similarity,
            "user_profiles": self.user_profiles,
            "n_users": self.n_users,
            "n_movies": self.n_movies,
        }
        joblib.dump(state, os.path.join(path, "content_model.joblib"))
        print(f"Content-based model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ContentBasedModel":
        """Load model from disk."""
        state = joblib.load(os.path.join(path, "content_model.joblib"))
        model = cls()
        model.genre_features = state["genre_features"]
        model.movie_similarity = state["movie_similarity"]
        model.user_profiles = state["user_profiles"]
        model.n_users = state["n_users"]
        model.n_movies = state["n_movies"]
        model._fitted = True
        return model

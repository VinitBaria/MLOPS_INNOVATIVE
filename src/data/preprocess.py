"""
Data Preprocessing Module
Handles feature engineering, train/test splitting, and data transformations.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.sparse import csr_matrix
import joblib

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")


class DataPreprocessor:
    """
    Preprocesses MovieLens data for recommendation models.
    
    Handles:
    - User/Movie ID encoding to contiguous indices
    - Train/test splitting (temporal or random)
    - User-item interaction matrix creation
    - Genre feature extraction
    - Data leakage prevention
    """

    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.rating_scaler = MinMaxScaler()
        self.n_users = 0
        self.n_movies = 0
        self.genre_columns = []
        self._fitted = False

    def fit_transform(
        self,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        test_size: float = 0.2,
        temporal_split: bool = True,
        random_state: int = 42,
    ) -> dict:
        """
        Fit preprocessor and transform data.
        
        Args:
            ratings: DataFrame with userId, movieId, rating, timestamp
            movies: DataFrame with movieId, title, genres, and genre flags
            test_size: Fraction of data for testing
            temporal_split: If True, split by time; if False, random split
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with train/test data and metadata
        """
        print("Preprocessing data...")

        # 1. Clean data
        ratings_clean = self._clean_ratings(ratings)

        # 2. Encode IDs
        ratings_clean["user_idx"] = self.user_encoder.fit_transform(ratings_clean["userId"])
        ratings_clean["movie_idx"] = self.movie_encoder.fit_transform(ratings_clean["movieId"])

        self.n_users = len(self.user_encoder.classes_)
        self.n_movies = len(self.movie_encoder.classes_)
        self._fitted = True

        print(f"  Encoded {self.n_users} users and {self.n_movies} movies")

        # 3. Split data
        if temporal_split:
            train_df, test_df = self._temporal_split(ratings_clean, test_size)
        else:
            train_df, test_df = train_test_split(
                ratings_clean, test_size=test_size, random_state=random_state
            )

        print(f"  Train: {len(train_df)} ratings, Test: {len(test_df)} ratings")

        # 4. Create interaction matrices
        train_matrix = self._create_interaction_matrix(train_df)
        test_matrix = self._create_interaction_matrix(test_df)

        # 5. Extract genre features
        genre_features = self._extract_genre_features(movies)

        # 6. Create user-movie mapping for evaluation
        train_user_items = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
        test_user_items = test_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()

        result = {
            "train_df": train_df,
            "test_df": test_df,
            "train_matrix": train_matrix,
            "test_matrix": test_matrix,
            "genre_features": genre_features,
            "train_user_items": train_user_items,
            "test_user_items": test_user_items,
            "n_users": self.n_users,
            "n_movies": self.n_movies,
            "movies": movies,
        }

        print("Preprocessing complete.")
        return result

    def transform_new_ratings(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Transform new ratings using fitted encoders."""
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")

        ratings_clean = self._clean_ratings(ratings)

        # Handle unseen users/movies
        known_users = set(self.user_encoder.classes_)
        known_movies = set(self.movie_encoder.classes_)

        mask = (
            ratings_clean["userId"].isin(known_users)
            & ratings_clean["movieId"].isin(known_movies)
        )

        ratings_known = ratings_clean[mask].copy()

        if len(ratings_known) < len(ratings_clean):
            n_dropped = len(ratings_clean) - len(ratings_known)
            print(f"  Dropped {n_dropped} ratings with unknown users/movies")

        ratings_known["user_idx"] = self.user_encoder.transform(ratings_known["userId"])
        ratings_known["movie_idx"] = self.movie_encoder.transform(ratings_known["movieId"])

        return ratings_known

    def _clean_ratings(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Clean raw ratings data."""
        df = ratings.copy()

        # Remove duplicates (keep latest)
        df = df.sort_values("timestamp").drop_duplicates(
            subset=["userId", "movieId"], keep="last"
        )

        # Remove invalid ratings
        df = df[(df["rating"] >= 1) & (df["rating"] <= 5)]

        # Remove null values
        df = df.dropna(subset=["userId", "movieId", "rating"])

        return df.reset_index(drop=True)

    def _temporal_split(
        self, df: pd.DataFrame, test_size: float
    ) -> tuple:
        """Split data by timestamp to prevent data leakage."""
        df_sorted = df.sort_values("timestamp")
        split_idx = int(len(df_sorted) * (1 - test_size))

        train = df_sorted.iloc[:split_idx].copy()
        test = df_sorted.iloc[split_idx:].copy()

        return train, test

    def _create_interaction_matrix(self, df: pd.DataFrame) -> csr_matrix:
        """Create a sparse user-item interaction matrix."""
        matrix = csr_matrix(
            (df["rating"].values, (df["user_idx"].values, df["movie_idx"].values)),
            shape=(self.n_users, self.n_movies),
        )
        return matrix

    def _extract_genre_features(self, movies: pd.DataFrame) -> np.ndarray:
        """Extract genre feature matrix for content-based filtering."""
        genre_cols = [
            "unknown", "Action", "Adventure", "Animation", "Children",
            "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
            "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
            "Sci-Fi", "Thriller", "War", "Western",
        ]

        self.genre_columns = genre_cols
        available_cols = [c for c in genre_cols if c in movies.columns]

        # Create genre matrix aligned with movie indices
        genre_matrix = np.zeros((self.n_movies, len(available_cols)))

        for _, row in movies.iterrows():
            movie_id = row["movieId"]
            if movie_id in self.movie_encoder.classes_:
                idx = self.movie_encoder.transform([movie_id])[0]
                genre_matrix[idx] = [row.get(g, 0) for g in available_cols]

        return genre_matrix

    def save(self, path: str = None):
        """Save preprocessor state."""
        if path is None:
            path = PROCESSED_DIR
        os.makedirs(path, exist_ok=True)

        state = {
            "user_encoder": self.user_encoder,
            "movie_encoder": self.movie_encoder,
            "n_users": self.n_users,
            "n_movies": self.n_movies,
            "genre_columns": self.genre_columns,
            "_fitted": self._fitted,
        }

        save_path = os.path.join(path, "preprocessor.joblib")
        joblib.dump(state, save_path)
        print(f"Preprocessor saved to {save_path}")

    @classmethod
    def load(cls, path: str = None) -> "DataPreprocessor":
        """Load preprocessor from saved state."""
        if path is None:
            path = PROCESSED_DIR

        load_path = os.path.join(path, "preprocessor.joblib")
        state = joblib.load(load_path)

        preprocessor = cls()
        preprocessor.user_encoder = state["user_encoder"]
        preprocessor.movie_encoder = state["movie_encoder"]
        preprocessor.n_users = state["n_users"]
        preprocessor.n_movies = state["n_movies"]
        preprocessor.genre_columns = state["genre_columns"]
        preprocessor._fitted = state["_fitted"]

        print(f"Preprocessor loaded from {load_path}")
        return preprocessor

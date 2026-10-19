"""
Model Logic Unit Tests
"""

import pytest
import numpy as np
from scipy.sparse import csr_matrix
from src.models.collaborative import CollaborativeFilteringModel
from src.models.content_based import ContentBasedModel
from src.models.hybrid import HybridRecommender


@pytest.fixture
def sample_data():
    # 5 users, 6 movies
    interactions = np.array([
        [5, 4, 0, 0, 1, 0],
        [0, 5, 4, 0, 0, 2],
        [3, 0, 0, 5, 4, 0],
        [0, 0, 5, 4, 0, 0],
        [1, 2, 0, 0, 5, 4],
    ])
    matrix = csr_matrix(interactions)

    # 6 movies x 4 genres
    genres = np.array([
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
    ])
    return matrix, genres


def test_collaborative_filtering(sample_data):
    matrix, _ = sample_data
    model = CollaborativeFilteringModel(n_factors=2)
    model.fit(matrix)

    pred = model.predict(user_idx=0, movie_idx=2)
    assert 1.0 <= pred <= 5.0

    recs = model.recommend(user_idx=0, n=3)
    assert len(recs) == 3
    assert all(isinstance(m_idx, int) for m_idx, _ in recs)


def test_content_based_model(sample_data):
    matrix, genres = sample_data
    model = ContentBasedModel()
    model.fit(matrix, genres)

    recs = model.recommend(user_idx=0, n=3)
    assert len(recs) == 3

    similar = model.get_similar_movies(movie_idx=0, n=2)
    assert len(similar) == 2


def test_hybrid_recommender(sample_data):
    matrix, genres = sample_data
    hybrid = HybridRecommender(n_factors=2, cf_weight=0.7, cb_weight=0.3)
    hybrid.fit(matrix, genres)

    recs = hybrid.recommend(user_idx=0, n=3)
    assert len(recs) == 3
    assert recs[0][1] >= recs[1][1]  # Ranked order check

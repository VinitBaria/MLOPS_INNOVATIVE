"""
Data Validation and Preprocessing Unit Tests
"""

import pytest
import pandas as pd
import numpy as np
from src.data.validate import DataValidator
from src.data.preprocess import DataPreprocessor


def test_data_validator_clean():
    validator = DataValidator()
    # Provide >= 10 users, >= 10 movies, >= 100 ratings
    users = list(range(1, 15)) * 8
    movies = list(range(10, 25)) * 8
    ratings = [4.0, 5.0, 3.5, 4.5, 3.0, 5.0, 4.0, 3.0] * 14
    df = pd.DataFrame({
        "userId": users[:105],
        "movieId": movies[:105],
        "rating": ratings[:105],
        "timestamp": pd.date_range("2024-01-01", periods=105),
    })
    report = validator.validate_ratings(df)
    assert report["status"] in ["PASS", "WARN"]


def test_data_validator_invalid_rating():
    validator = DataValidator()
    df = pd.DataFrame({
        "userId": [1, 2],
        "movieId": [10, 20],
        "rating": [6.5, -1.0],  # Out of bounds
    })
    report = validator.validate_ratings(df)
    assert report["status"] == "FAIL"


def test_validate_new_rating():
    validator = DataValidator()
    valid = validator.validate_new_rating(user_id=1, movie_id=42, rating=4.5)
    assert valid["valid"] is True

    invalid = validator.validate_new_rating(user_id=-1, movie_id=42, rating=10.0)
    assert invalid["valid"] is False
    assert len(invalid["issues"]) == 2

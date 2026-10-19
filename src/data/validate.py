"""
Data Validation Module
Validates data quality before training and during inference.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any


class DataValidator:
    """
    Validates MovieLens data for training and serving pipelines.
    
    Checks:
    - Missing values
    - Duplicate records
    - Invalid ratings
    - Schema conformity
    - Distribution anomalies
    """

    def __init__(self):
        self.validation_results: List[Dict[str, Any]] = []

    def validate_ratings(self, ratings: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate ratings dataset.
        
        Args:
            ratings: DataFrame with userId, movieId, rating, timestamp
            
        Returns:
            Dictionary with validation results and status
        """
        self.validation_results = []
        
        # Check required columns
        required_cols = ["userId", "movieId", "rating"]
        self._check_required_columns(ratings, required_cols)

        # Check for null values
        self._check_nulls(ratings, "ratings")

        # Check rating range
        self._check_rating_range(ratings)

        # Check for duplicates
        self._check_duplicates(ratings, ["userId", "movieId"])

        # Check minimum data requirements
        self._check_minimum_data(ratings)

        # Check for data anomalies
        self._check_rating_distribution(ratings)

        # Compute overall status
        passed = all(r["status"] == "PASS" for r in self.validation_results)

        report = {
            "status": "PASS" if passed else "FAIL",
            "checks": self.validation_results,
            "summary": {
                "total_checks": len(self.validation_results),
                "passed": sum(1 for r in self.validation_results if r["status"] == "PASS"),
                "failed": sum(1 for r in self.validation_results if r["status"] == "FAIL"),
                "warnings": sum(1 for r in self.validation_results if r["status"] == "WARN"),
            },
        }

        return report

    def validate_movies(self, movies: pd.DataFrame) -> Dict[str, Any]:
        """Validate movies dataset."""
        self.validation_results = []

        required_cols = ["movieId", "title"]
        self._check_required_columns(movies, required_cols)
        self._check_nulls(movies[["movieId", "title"]], "movies")
        self._check_unique_ids(movies, "movieId")

        passed = all(r["status"] == "PASS" for r in self.validation_results)

        return {
            "status": "PASS" if passed else "FAIL",
            "checks": self.validation_results,
        }

    def validate_new_rating(self, user_id: int, movie_id: int, rating: float) -> Dict[str, Any]:
        """Validate a single new rating before ingestion."""
        issues = []

        if not isinstance(user_id, (int, np.integer)) or user_id < 1:
            issues.append("Invalid user_id: must be a positive integer")

        if not isinstance(movie_id, (int, np.integer)) or movie_id < 1:
            issues.append("Invalid movie_id: must be a positive integer")

        if not isinstance(rating, (int, float, np.integer, np.floating)):
            issues.append("Invalid rating: must be a number")
        elif rating < 1 or rating > 5:
            issues.append(f"Invalid rating {rating}: must be between 1 and 5")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
        }

    def _check_required_columns(self, df: pd.DataFrame, required: List[str]):
        missing = [c for c in required if c not in df.columns]
        self.validation_results.append({
            "check": "required_columns",
            "status": "PASS" if not missing else "FAIL",
            "detail": f"Missing columns: {missing}" if missing else "All required columns present",
        })

    def _check_nulls(self, df: pd.DataFrame, name: str):
        null_counts = df.isnull().sum()
        has_nulls = null_counts.sum() > 0
        self.validation_results.append({
            "check": f"null_values_{name}",
            "status": "WARN" if has_nulls else "PASS",
            "detail": f"Null counts: {null_counts.to_dict()}" if has_nulls else "No null values",
        })

    def _check_rating_range(self, df: pd.DataFrame):
        out_of_range = ((df["rating"] < 1) | (df["rating"] > 5)).sum()
        self.validation_results.append({
            "check": "rating_range",
            "status": "FAIL" if out_of_range > 0 else "PASS",
            "detail": f"{out_of_range} ratings out of [1, 5] range" if out_of_range else "All ratings in valid range",
        })

    def _check_duplicates(self, df: pd.DataFrame, subset: List[str]):
        n_dupes = df.duplicated(subset=subset).sum()
        self.validation_results.append({
            "check": "duplicates",
            "status": "WARN" if n_dupes > 0 else "PASS",
            "detail": f"{n_dupes} duplicate records" if n_dupes else "No duplicates",
        })

    def _check_unique_ids(self, df: pd.DataFrame, id_col: str):
        n_dupes = df[id_col].duplicated().sum()
        self.validation_results.append({
            "check": f"unique_{id_col}",
            "status": "FAIL" if n_dupes > 0 else "PASS",
            "detail": f"{n_dupes} duplicate {id_col} values" if n_dupes else f"All {id_col} values unique",
        })

    def _check_minimum_data(self, df: pd.DataFrame):
        n_users = df["userId"].nunique()
        n_movies = df["movieId"].nunique()
        n_ratings = len(df)

        issues = []
        if n_users < 10:
            issues.append(f"Only {n_users} users (minimum 10)")
        if n_movies < 10:
            issues.append(f"Only {n_movies} movies (minimum 10)")
        if n_ratings < 100:
            issues.append(f"Only {n_ratings} ratings (minimum 100)")

        self.validation_results.append({
            "check": "minimum_data",
            "status": "FAIL" if issues else "PASS",
            "detail": "; ".join(issues) if issues else f"{n_users} users, {n_movies} movies, {n_ratings} ratings",
        })

    def _check_rating_distribution(self, df: pd.DataFrame):
        """Check for suspicious rating distributions."""
        rating_counts = df["rating"].value_counts(normalize=True)

        # Flag if any single rating value exceeds 80%
        max_pct = rating_counts.max()
        dominant_rating = rating_counts.idxmax()

        self.validation_results.append({
            "check": "rating_distribution",
            "status": "WARN" if max_pct > 0.8 else "PASS",
            "detail": (
                f"Rating {dominant_rating} dominates with {max_pct:.1%}"
                if max_pct > 0.8
                else f"Rating distribution looks normal (max {max_pct:.1%})"
            ),
        })


def print_validation_report(report: Dict[str, Any]):
    """Print a human-readable validation report."""
    status_icon = "✅" if report["status"] == "PASS" else "❌"
    print(f"\n{status_icon} Validation Status: {report['status']}")
    print("-" * 50)

    for check in report["checks"]:
        icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}.get(check["status"], "❓")
        print(f"  {icon} {check['check']}: {check['detail']}")

    if "summary" in report:
        s = report["summary"]
        print(f"\n  Total: {s['total_checks']} | Passed: {s['passed']} | "
              f"Failed: {s['failed']} | Warnings: {s['warnings']}")

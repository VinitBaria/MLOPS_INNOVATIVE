"""
API Integration Tests using FastAPI TestClient
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "db_connected" in data


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "recommendation_requests_total" in response.text


def test_model_info_endpoint():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data


def test_recommendation_endpoint():
    response = client.get("/recommend/1?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert "recommendations" in data
    assert "latency_ms" in data


def test_submit_rating():
    payload = {
        "user_id": 1,
        "movie_id": 1,
        "rating": 4.5
    }
    response = client.post("/ratings", json=payload)
    # If movie 1 exists or not
    assert response.status_code in [200, 404]

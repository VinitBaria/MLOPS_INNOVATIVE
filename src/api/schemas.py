"""
Pydantic Schemas for FastAPI Request & Response Validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class MovieItem(BaseModel):
    movie_id: int
    title: str
    genres: str
    predicted_rating: Optional[float] = None
    similarity_score: Optional[float] = None


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[MovieItem]
    model_version: str
    model_type: str
    latency_ms: float


class RatingCreate(BaseModel):
    user_id: int = Field(..., ge=1, description="Unique positive integer user ID")
    movie_id: int = Field(..., ge=1, description="Movie ID from database")
    rating: float = Field(..., ge=1.0, le=5.0, description="Rating from 1.0 to 5.0")


class RatingResponse(BaseModel):
    status: str
    message: str
    user_id: int
    movie_id: int
    rating: float
    total_new_ratings: int
    retraining_triggered: bool = False


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    model_loaded: bool
    model_version: str
    db_connected: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    best_model_type: str
    ndcg_at_10: Optional[float] = None
    recall_at_10: Optional[float] = None
    rmse: Optional[float] = None
    training_timestamp: Optional[str] = None

"""
SQLAlchemy ORM Data Models for Movie Recommender
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from src.database.db import Base


class Movie(Base):
    __tablename__ = "movies"

    id = Column(Integer, primary_key=True, index=True)
    movie_id = Column(Integer, unique=True, index=True, nullable=False)
    title = Column(String(255), nullable=False)
    release_date = Column(String(50), nullable=True)
    genres = Column(String(255), nullable=False)

    ratings = relationship("Rating", back_populates="movie", cascade="all, delete-orphan")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, unique=True, index=True, nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String(10), nullable=True)
    occupation = Column(String(50), nullable=True)
    zip_code = Column(String(20), nullable=True)

    ratings = relationship("Rating", back_populates="user", cascade="all, delete-orphan")


class Rating(Base):
    __tablename__ = "ratings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False, index=True)
    movie_id = Column(Integer, ForeignKey("movies.movie_id"), nullable=False, index=True)
    rating = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    is_new = Column(Integer, default=1)  # 1 for newly ingested ratings since last training

    user = relationship("User", back_populates="ratings")
    movie = relationship("Movie", back_populates="ratings")


class RecommendationLog(Base):
    __tablename__ = "recommendation_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    recommended_movies = Column(Text, nullable=False)  # JSON or comma-separated movie IDs
    model_version = Column(String(50), nullable=False)
    latency_ms = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class DriftLog(Base):
    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), nullable=False)
    drift_score = Column(Float, nullable=False)
    p_value = Column(Float, nullable=True)
    drift_detected = Column(Integer, default=0)
    checked_at = Column(DateTime, default=datetime.utcnow)

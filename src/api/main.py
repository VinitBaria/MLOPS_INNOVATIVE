"""
FastAPI Model Serving API for Movie Recommendation System
Provides endpoints for personalized recommendations, rating ingestion,
Prometheus observability, health checks, and retraining triggers.
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, Query, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.database.db import get_db, engine, Base
from src.database.models import Movie, User, Rating, RecommendationLog
from src.data.preprocess import DataPreprocessor
from src.models.hybrid import HybridRecommender
from src.api.schemas import (
    RecommendationResponse,
    MovieItem,
    RatingCreate,
    RatingResponse,
    HealthResponse,
    ModelInfoResponse,
)
from src.monitoring.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    MODEL_INFERENCE_TIME,
    RECOMMENDATIONS_SERVED,
    RATINGS_INGESTED,
    ACTIVE_MODEL_VERSION,
    MODEL_NDCG_SCORE,
    MODEL_RECALL_SCORE,
)
from src.training.retrain import run_automated_retraining

# Ensure DB schema is created
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Movie Recommendation System API",
    description="End-to-end MLOps Movie Recommendation Service with Real-Time Inference & Observability",
    version="1.0.0",
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
MODEL = None
PREPROCESSOR = None
MODEL_METADATA = {}
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "artifacts", "production")
METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "artifacts", "metadata.json")
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD_NEW_RATINGS", "50"))


def load_model_and_preprocessor():
    """Load model artifact and preprocessor into memory."""
    global MODEL, PREPROCESSOR, MODEL_METADATA
    try:
        PREPROCESSOR = DataPreprocessor.load()
        MODEL = HybridRecommender.load(MODEL_PATH)

        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r") as f:
                MODEL_METADATA = json.load(f)

        best_m = MODEL_METADATA.get("best_model", "hybrid_recommender")
        ndcg = MODEL_METADATA.get("metrics", {}).get(best_m, {}).get("ndcg_at_10", 0.0)
        recall = MODEL_METADATA.get("metrics", {}).get(best_m, {}).get("recall_at_10", 0.0)

        ACTIVE_MODEL_VERSION.labels(model_name="hybrid-recommender").set(1)
        MODEL_NDCG_SCORE.set(ndcg)
        MODEL_RECALL_SCORE.set(recall)
        print("Production recommendation model loaded successfully.")
    except Exception as e:
        print(f"Notice: Model artifacts not loaded yet ({e}). API running in fallback mode.")


@app.on_event("startup")
def startup_event():
    load_model_and_preprocessor()


# ─── Health & Observability ───

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check(db: Session = Depends(get_db)):
    """Service health and model readiness probe."""
    db_ok = True
    try:
        db.query(Movie).first()
    except Exception:
        db_ok = False

    return HealthResponse(
        status="healthy" if (db_ok and MODEL is not None) else "degraded",
        timestamp=datetime.utcnow(),
        model_loaded=MODEL is not None,
        model_version=MODEL_METADATA.get("model_version", "v1.0.0"),
        db_connected=db_ok,
    )


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus metrics endpoint scraped by Prometheus."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["MLOps"])
def model_info():
    """Metadata regarding current deployed model version, hyperparameters, and quality metrics."""
    best_m = MODEL_METADATA.get("best_model", "hybrid_recommender")
    metrics_dict = MODEL_METADATA.get("metrics", {}).get(best_m, {})

    return ModelInfoResponse(
        model_name="hybrid-movie-recommender",
        model_version=MODEL_METADATA.get("model_version", "production-v1"),
        best_model_type=best_m,
        ndcg_at_10=metrics_dict.get("ndcg_at_10"),
        recall_at_10=metrics_dict.get("recall_at_10"),
        rmse=metrics_dict.get("rmse"),
        training_timestamp=MODEL_METADATA.get("timestamp"),
    )


# ─── Recommendation Endpoints ───

@app.get("/recommend/{user_id}", response_model=RecommendationResponse, tags=["Recommendations"])
def get_recommendations(
    user_id: int,
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """
    Returns personalized top-N movie recommendations for a user.
    Leverages collaborative & content-based hybrid filtering with cold-start fallback.
    """
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="recommend", status="received").inc()

    # User watched movies
    user_ratings = db.query(Rating).filter(Rating.user_id == user_id).all()
    watched_movie_ids = {r.movie_id for r in user_ratings}

    recommendations: List[MovieItem] = []
    model_type = "hybrid"

    if MODEL is not None and PREPROCESSOR is not None:
        try:
            # Map raw IDs to internal indices
            user_idx = (
                PREPROCESSOR.user_encoder.transform([user_id])[0]
                if user_id in PREPROCESSOR.user_encoder.classes_
                else 999999  # cold start user
            )

            exclude_indices = set()
            for m_id in watched_movie_ids:
                if m_id in PREPROCESSOR.movie_encoder.classes_:
                    exclude_indices.add(PREPROCESSOR.movie_encoder.transform([m_id])[0])

            # Model inference
            t_inf_start = time.time()
            raw_recs = MODEL.recommend(user_idx=user_idx, n=limit, exclude_watched=exclude_indices)
            MODEL_INFERENCE_TIME.labels(model_type="hybrid").observe(time.time() - t_inf_start)

            # Map movie indices back to database movies
            for movie_idx, score in raw_recs:
                if movie_idx < len(PREPROCESSOR.movie_encoder.classes_):
                    m_id = int(PREPROCESSOR.movie_encoder.inverse_transform([movie_idx])[0])
                    movie_obj = db.query(Movie).filter(Movie.movie_id == m_id).first()
                    if movie_obj:
                        recommendations.append(
                            MovieItem(
                                movie_id=movie_obj.movie_id,
                                title=movie_obj.title,
                                genres=movie_obj.genres,
                                predicted_rating=round(float(score), 2),
                            )
                        )
        except Exception as e:
            print(f"Inference error: {e}")
            recommendations = []

    # Database fallback if model not loaded or returned empty
    if not recommendations:
        model_type = "popularity_fallback"
        popular_movies = (
            db.query(Movie)
            .filter(~Movie.movie_id.in_(watched_movie_ids) if watched_movie_ids else True)
            .limit(limit)
            .all()
        )
        for m in popular_movies:
            recommendations.append(
                MovieItem(
                    movie_id=m.movie_id,
                    title=m.title,
                    genres=m.genres,
                    predicted_rating=4.0,
                )
            )

    latency_ms = (time.time() - start_time) * 1000.0
    REQUEST_LATENCY.labels(endpoint="recommend").observe(latency_ms / 1000.0)
    RECOMMENDATIONS_SERVED.inc(len(recommendations))
    REQUEST_COUNT.labels(endpoint="recommend", status="success").inc()

    # Log recommendation event
    try:
        log_entry = RecommendationLog(
            user_id=user_id,
            recommended_movies=",".join(str(r.movie_id) for r in recommendations),
            model_version=MODEL_METADATA.get("model_version", "v1.0.0"),
            latency_ms=latency_ms,
        )
        db.add(log_entry)
        db.commit()
    except Exception:
        db.rollback()

    return RecommendationResponse(
        user_id=user_id,
        recommendations=recommendations,
        model_version=MODEL_METADATA.get("model_version", "v1.0.0"),
        model_type=model_type,
        latency_ms=round(latency_ms, 2),
    )


# ─── Movie Catalog Endpoints ───

@app.get("/movies", response_model=List[MovieItem], tags=["Catalog"])
def list_movies(
    query: Optional[str] = None,
    genre: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
):
    """Search and browse movie catalog."""
    q = db.query(Movie)
    if query:
        q = q.filter(Movie.title.ilike(f"%{query}%"))
    if genre:
        q = q.filter(Movie.genres.ilike(f"%{genre}%"))
    movies = q.offset(skip).limit(limit).all()

    return [
        MovieItem(
            movie_id=m.movie_id,
            title=m.title,
            genres=m.genres,
        )
        for m in movies
    ]


@app.get("/movies/{movie_id}/similar", response_model=List[MovieItem], tags=["Catalog"])
def get_similar_movies(movie_id: int, limit: int = 6, db: Session = Depends(get_db)):
    """Return movies with similar content/genres."""
    target_movie = db.query(Movie).filter(Movie.movie_id == movie_id).first()
    if not target_movie:
        raise HTTPException(status_code=404, detail="Movie not found")

    similar_movies = []
    if MODEL is not None and PREPROCESSOR is not None and hasattr(MODEL, "get_similar_movies"):
        try:
            if movie_id in PREPROCESSOR.movie_encoder.classes_:
                m_idx = PREPROCESSOR.movie_encoder.transform([movie_id])[0]
                sim_pairs = MODEL.get_similar_movies(m_idx, n=limit)
                for s_idx, sim_score in sim_pairs:
                    raw_id = int(PREPROCESSOR.movie_encoder.inverse_transform([s_idx])[0])
                    m_obj = db.query(Movie).filter(Movie.movie_id == raw_id).first()
                    if m_obj:
                        similar_movies.append(
                            MovieItem(
                                movie_id=m_obj.movie_id,
                                title=m_obj.title,
                                genres=m_obj.genres,
                                similarity_score=round(float(sim_score), 2),
                            )
                        )
        except Exception:
            pass

    # Fallback to genre query
    if not similar_movies:
        primary_genre = target_movie.genres.split("|")[0] if target_movie.genres else "Drama"
        sim_objs = (
            db.query(Movie)
            .filter(Movie.genres.ilike(f"%{primary_genre}%"), Movie.movie_id != movie_id)
            .limit(limit)
            .all()
        )
        for m_obj in sim_objs:
            similar_movies.append(
                MovieItem(
                    movie_id=m_obj.movie_id,
                    title=m_obj.title,
                    genres=m_obj.genres,
                    similarity_score=0.85,
                )
            )

    return similar_movies


# ─── Rating Ingestion & Automated Retraining ───

@app.post("/ratings", response_model=RatingResponse, tags=["Ratings"])
def submit_rating(
    rating_data: RatingCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Submits a user rating for a movie.
    Persists to database and checks retraining threshold.
    """
    movie = db.query(Movie).filter(Movie.movie_id == rating_data.movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail=f"Movie {rating_data.movie_id} does not exist.")

    # Upsert rating
    existing_rating = (
        db.query(Rating)
        .filter(Rating.user_id == rating_data.user_id, Rating.movie_id == rating_data.movie_id)
        .first()
    )

    if existing_rating:
        existing_rating.rating = rating_data.rating
        existing_rating.is_new = 1
    else:
        new_rating = Rating(
            user_id=rating_data.user_id,
            movie_id=rating_data.movie_id,
            rating=rating_data.rating,
            is_new=1,
        )
        db.add(new_rating)

    db.commit()
    RATINGS_INGESTED.inc()

    # Count pending new ratings
    pending_count = db.query(Rating).filter(Rating.is_new == 1).count()
    retrain_triggered = False

    if pending_count >= RETRAIN_THRESHOLD:
        print(f"Retraining threshold reached ({pending_count} >= {RETRAIN_THRESHOLD}). Triggering background retraining.")
        background_tasks.add_task(run_automated_retraining)
        retrain_triggered = True

    return RatingResponse(
        status="success",
        message="Rating recorded successfully.",
        user_id=rating_data.user_id,
        movie_id=rating_data.movie_id,
        rating=rating_data.rating,
        total_new_ratings=pending_count,
        retraining_triggered=retrain_triggered,
    )


@app.post("/model/retrain", tags=["MLOps"])
def trigger_retrain(background_tasks: BackgroundTasks):
    """Manually trigger the automated retraining and quality gate pipeline."""
    background_tasks.add_task(run_automated_retraining, force=True)
    return {"status": "accepted", "message": "Automated retraining pipeline started in background."}


@app.post("/model/reload", tags=["MLOps"])
def reload_model():
    """Dynamically reloads production model artifact from disk without server restart."""
    load_model_and_preprocessor()
    return {"status": "success", "message": "Model reloaded into memory."}


# ─── Static UI files mount ───
STATIC_DIR = os.path.join(PROJECT_ROOT, "frontend", "build")
if os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")

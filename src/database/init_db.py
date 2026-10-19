"""
Database Initializer & Seed Script
Creates tables and seeds SQLite/PostgreSQL with MovieLens movies and sample users.
"""

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.database.db import engine, Base, SessionLocal
from src.database.models import Movie, User, Rating
from src.data.download import download_movielens, load_movies, load_users, load_ratings


def init_database(seed_data: bool = True, limit_ratings: int = 10000):
    """
    Creates tables and seeds database with MovieLens data.
    """
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables initialized successfully.")

    if not seed_data:
        return

    db = SessionLocal()
    try:
        # Check if already seeded
        if db.query(Movie).count() > 0:
            print("Database already contains data. Skipping seed.")
            return

        print("Seeding database with MovieLens data...")
        data_dir = download_movielens()
        movies_df = load_movies(data_dir)
        users_df = load_users(data_dir)
        ratings_df = load_ratings(data_dir)

        # Insert Movies
        movie_records = []
        for _, row in movies_df.iterrows():
            movie_records.append(
                Movie(
                    movie_id=int(row["movieId"]),
                    title=str(row["title"]),
                    release_date=str(row["release_date"]) if pd_not_na(row["release_date"]) else None,
                    genres=str(row["genres"]),
                )
            )
        db.bulk_save_objects(movie_records)
        db.commit()
        print(f"Inserted {len(movie_records)} movies.")

        # Insert Users
        user_records = []
        for _, row in users_df.iterrows():
            user_records.append(
                User(
                    user_id=int(row["userId"]),
                    age=int(row["age"]) if pd_not_na(row["age"]) else None,
                    gender=str(row["gender"]) if pd_not_na(row["gender"]) else None,
                    occupation=str(row["occupation"]) if pd_not_na(row["occupation"]) else None,
                    zip_code=str(row["zip_code"]) if pd_not_na(row["zip_code"]) else None,
                )
            )
        db.bulk_save_objects(user_records)
        db.commit()
        print(f"Inserted {len(user_records)} users.")

        # Insert initial ratings (sample if limit specified to ensure quick seeding)
        if limit_ratings and len(ratings_df) > limit_ratings:
            ratings_sample = ratings_df.head(limit_ratings)
        else:
            ratings_sample = ratings_df

        rating_records = []
        for _, row in ratings_sample.iterrows():
            rating_records.append(
                Rating(
                    user_id=int(row["userId"]),
                    movie_id=int(row["movieId"]),
                    rating=float(row["rating"]),
                    created_at=row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else None,
                    is_new=0,
                )
            )
        db.bulk_save_objects(rating_records)
        db.commit()
        print(f"Inserted {len(rating_records)} baseline ratings.")

    except Exception as e:
        db.rollback()
        print(f"Error seeding database: {e}")
        raise
    finally:
        db.close()


def pd_not_na(val):
    return val is not None and str(val) != "nan" and str(val) != ""


if __name__ == "__main__":
    init_database()

"""
Database Connection & Session Management
Supports PostgreSQL with automated SQLite fallback for frictionless local testing.
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Load .env file if present
load_dotenv()

# Database configuration from environment
DATABASE_URL = os.getenv("DATABASE_URL")
POSTGRES_USER = os.getenv("POSTGRES_USER", "movie_rec_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "movie_rec_pass_2024")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "movie_rec_db")
USE_SQLITE = os.getenv("USE_SQLITE", "false").lower() == "true"

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
os.makedirs(DB_DIR, exist_ok=True)
SQLITE_PATH = os.path.join(DB_DIR, "movies.db")

if not DATABASE_URL:
    DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# SQLAlchemy requires postgresql+psycopg2:// when psycopg2 is installed
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://") and not DATABASE_URL.startswith("postgresql+"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

try:
    if USE_SQLITE:
        raise ValueError("Configured for SQLite explicitly.")
    # Attempt Postgres connection
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args={"connect_timeout": 3})
    with engine.connect() as conn:
        pass
    print("Connected to PostgreSQL database.")
except Exception:
    # Fallback to local SQLite database for zero-friction standalone run
    DATABASE_URL = f"sqlite:///{SQLITE_PATH}"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    print(f"PostgreSQL unavailable. Using local SQLite database at: {SQLITE_PATH}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency for FastAPI route handlers."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

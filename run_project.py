"""
Master MLOps Runner Script
Provides interactive CLI to run data download, training, testing, or API serving.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="🎬 CineMatch MLOps Project Manager")
    parser.add_argument(
        "action",
        choices=["setup", "train", "test", "serve", "all"],
        default="all",
        nargs="?",
        help="Action to perform: 'setup' (seed DB), 'train' (run ML pipeline), 'test' (run pytest), 'serve' (start API + Web UI), or 'all' (run complete workflow)",
    )
    args = parser.parse_args()

    if args.action in ["setup", "all"]:
        print("\n[Step 1/3] Initializing Database & Downloading MovieLens...")
        from src.database.init_db import init_database
        init_database(seed_data=True, limit_ratings=15000)

    if args.action in ["train", "all"]:
        print("\n[Step 2/3] Training Models (CF, CB, Hybrid) & Computing Metrics...")
        from src.training.train import train_pipeline
        train_pipeline(n_factors=20, use_mlflow=False)

    if args.action == "test":
        print("\nRunning automated Pytest suite...")
        import pytest
        pytest.main(["tests/", "-v", "--disable-warnings"])

    if args.action in ["serve", "all"]:
        print("\n[Step 3/3] Starting FastAPI Recommender Service & Web UI...")
        print("  Web App & Recommendations UI: http://localhost:8000")
        print("  Interactive Swagger API Docs: http://localhost:8000/docs")
        print("  Prometheus Metrics Endpoint:  http://localhost:8000/metrics")
        print("Press Ctrl+C to stop server.\n")

        import uvicorn
        uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()

"""
MovieLens Dataset Downloader
Downloads and extracts the MovieLens 100K dataset for the recommendation system.
"""

import os
import zipfile
import requests
import pandas as pd

# MovieLens 100K dataset URL
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def download_movielens(url: str = MOVIELENS_URL, dest_dir: str = RAW_DIR) -> str:
    """
    Download the MovieLens 100K dataset.
    
    Args:
        url: URL to download from
        dest_dir: Directory to save the zip file
        
    Returns:
        Path to the extracted dataset directory
    """
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "ml-100k.zip")
    extract_dir = os.path.join(dest_dir, "ml-100k")

    # Skip download if already extracted
    if os.path.exists(extract_dir):
        print(f"Dataset already exists at {extract_dir}")
        return extract_dir

    # Download
    print(f"Downloading MovieLens 100K from {url}...")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                print(f"\rProgress: {pct:.1f}%", end="", flush=True)

    print("\nDownload complete.")

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_dir)

    # Clean up zip
    os.remove(zip_path)
    print(f"Dataset extracted to {extract_dir}")

    return extract_dir


def load_ratings(data_dir: str = None) -> pd.DataFrame:
    """
    Load the ratings data from MovieLens 100K.
    
    Returns:
        DataFrame with columns: userId, movieId, rating, timestamp
    """
    if data_dir is None:
        data_dir = os.path.join(RAW_DIR, "ml-100k")

    ratings_path = os.path.join(data_dir, "u.data")

    if not os.path.exists(ratings_path):
        raise FileNotFoundError(
            f"Ratings file not found at {ratings_path}. "
            "Run download_movielens() first."
        )

    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["userId", "movieId", "rating", "timestamp"],
        encoding="latin-1",
    )

    # Convert timestamp to datetime
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")

    print(f"Loaded {len(ratings)} ratings from {ratings['userId'].nunique()} users "
          f"on {ratings['movieId'].nunique()} movies")

    return ratings


def load_movies(data_dir: str = None) -> pd.DataFrame:
    """
    Load the movie metadata from MovieLens 100K.
    
    Returns:
        DataFrame with columns: movieId, title, release_date, genres (+ individual genre columns)
    """
    if data_dir is None:
        data_dir = os.path.join(RAW_DIR, "ml-100k")

    movies_path = os.path.join(data_dir, "u.item")

    if not os.path.exists(movies_path):
        raise FileNotFoundError(
            f"Movies file not found at {movies_path}. "
            "Run download_movielens() first."
        )

    genre_names = [
        "unknown", "Action", "Adventure", "Animation", "Children",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
        "Sci-Fi", "Thriller", "War", "Western",
    ]

    # MovieLens 100K u.item has: movieId | title | release_date | video_release | url | genre_flags...
    movies = pd.read_csv(
        movies_path,
        sep="|",
        names=["movieId", "title", "release_date", "video_release_date", "url"] + genre_names,
        encoding="latin-1",
    )

    # Drop unnecessary columns
    movies = movies.drop(columns=["video_release_date", "url"])

    # Create a combined genres string
    def get_genres(row):
        return "|".join([g for g in genre_names if row[g] == 1])

    movies["genres"] = movies.apply(get_genres, axis=1)

    print(f"Loaded {len(movies)} movies with {len(genre_names)} genre categories")

    return movies


def load_users(data_dir: str = None) -> pd.DataFrame:
    """
    Load user demographics from MovieLens 100K.
    
    Returns:
        DataFrame with columns: userId, age, gender, occupation, zip_code
    """
    if data_dir is None:
        data_dir = os.path.join(RAW_DIR, "ml-100k")

    users_path = os.path.join(data_dir, "u.user")

    if not os.path.exists(users_path):
        raise FileNotFoundError(
            f"Users file not found at {users_path}. "
            "Run download_movielens() first."
        )

    users = pd.read_csv(
        users_path,
        sep="|",
        names=["userId", "age", "gender", "occupation", "zip_code"],
        encoding="latin-1",
    )

    print(f"Loaded {len(users)} users")

    return users


if __name__ == "__main__":
    # Download and load as a quick test
    extract_dir = download_movielens()
    ratings = load_ratings(extract_dir)
    movies = load_movies(extract_dir)
    users = load_users(extract_dir)

    print("\n--- Dataset Summary ---")
    print(f"Ratings: {len(ratings)}")
    print(f"Movies:  {len(movies)}")
    print(f"Users:   {len(users)}")
    print(f"\nRating distribution:\n{ratings['rating'].value_counts().sort_index()}")

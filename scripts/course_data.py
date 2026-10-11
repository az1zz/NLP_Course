"""Locate the exact public datasets referenced by the instructor."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("KAGGLEHUB_CACHE", str(ROOT / ".cache" / "kagglehub"))

DATASETS = {
    "imdb": ("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews", 1),
    "apple": ("seriousran/appletwittersentimenttexts", 1),
    "pakistan": ("adizafar/large-random-tweets-from-pakistan", 2),
    "disneyland": ("arushchillar/disneyland-reviews", 1),
    "amazon": ("PromptCloudHQ/amazon-reviews-unlocked-mobile-phones", 1),
    "news": ("clmentbisaillon/fake-and-real-news-dataset", 1),
}


def dataset_file(key, filename):
    import kagglehub

    handle, version = DATASETS[key]
    folder = Path(kagglehub.dataset_download(f"{handle}/versions/{version}"))
    matches = list(folder.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {filename} in {folder}; found {len(matches)}")
    return matches[0]


if __name__ == "__main__":
    filenames = {
        "imdb": "IMDB Dataset.csv",
        "apple": "apple-twitter-sentiment-texts.csv",
        "pakistan": "Random Tweets from Pakistan- Cleaned- Anonymous.csv",
        "disneyland": "DisneylandReviews.csv",
        "amazon": "Amazon_Unlocked_Mobile.csv",
        "news": "True.csv",
    }
    for key, filename in filenames.items():
        path = dataset_file(key, filename)
        print(f"{key}: {path.name} ({path.stat().st_size:,} bytes)", flush=True)

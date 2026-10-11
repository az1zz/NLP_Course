"""Download the NLTK resources needed by the notebooks."""
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
destination = ROOT / ".cache" / "nltk_data"
destination.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NLTK_DATA", str(destination))

import nltk

for resource in ["punkt", "punkt_tab", "wordnet", "stopwords",
                 "averaged_perceptron_tagger_eng"]:
    for attempt in range(3):
        try:
            if not nltk.download(resource, download_dir=str(destination), quiet=True, raise_on_error=True):
                raise RuntimeError(f"Failed to download NLTK resource {resource}")
            break
        except (ValueError, OSError):
            if attempt == 2:
                raise
            print(f"Retrying NLTK resource {resource} after a connection failure", flush=True)
            time.sleep(2)
    print(f"NLTK resource ready: {resource}", flush=True)

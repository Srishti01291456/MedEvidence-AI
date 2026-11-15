from datasets import load_dataset
import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset directories
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_medquad():

    print("Downloading MedQuAD dataset...")

    dataset = load_dataset(
        "lavita/MedQuAD",
        split="train"
    )

    df = dataset.to_pandas()

    print("\nDataset downloaded successfully!")

    print("Total records:", len(df))

    print("\nOriginal columns:")
    print(df.columns.tolist())

    # Save original dataset without deleting records
    raw_path = RAW_DIR / "medquad_raw.csv"
    df.to_csv(raw_path, index=False)

    print("\nRaw dataset saved:", raw_path)

    return df


if __name__ == "__main__":
    df = download_medquad()

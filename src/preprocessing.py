
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_DIR / "data" / "raw" / "medquad_raw.csv"

OUTPUT_PATH = (
    BASE_DIR / "data" / "processed" / "medquad_clean.csv"
)


def valid_url(url):

    if pd.isna(url):
        return False

    url = str(url).strip()

    try:
        parsed = urlparse(url)

        return (
            parsed.scheme in ["http", "https"]
            and bool(parsed.netloc)
        )

    except Exception:
        return False


def clean_dataset():

    print("Loading raw dataset...")

    df = pd.read_csv(RAW_PATH)

    initial_count = len(df)

    # Convert missing text values into empty strings
    text_columns = [
        "question",
        "answer",
        "document_source",
        "document_url"
    ]

    for column in text_columns:
        df[column] = df[column].fillna("").astype(str).str.strip()

    print("\nInitial records:", initial_count)

    # Remove records without questions
    df = df[df["question"] != ""].copy()

    # Remove records without answers
    df = df[df["answer"] != ""].copy()

    # Remove duplicate question-answer pairs
    df = df.drop_duplicates(
        subset=["question", "answer"]
    ).copy()

    # Preserve original source metadata
    df["source_name"] = df["document_source"]
    df["source_url"] = df["document_url"]

    # Check URL format without inventing or modifying URLs
    df["source_url_valid"] = (
        df["source_url"].apply(valid_url)
    )

    # Create combined text for future retrieval
    df["retrieval_text"] = (
        "Question: " + df["question"]
        + "\nAnswer: " + df["answer"]
    )

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n========== CLEANING REPORT ==========")

    print("Initial records:", initial_count)

    print("Final records:", len(df))

    print("Removed records:", initial_count - len(df))

    print(
        "Missing source names:",
        (df["source_name"] == "").sum()
    )

    print(
        "Missing source URLs:",
        (df["source_url"] == "").sum()
    )

    print(
        "Valid URL formats:",
        df["source_url_valid"].sum()
    )

    print(
        "Invalid or missing URL formats:",
        (~df["source_url_valid"]).sum()
    )

    print("\nFinal columns:")
    print(df.columns.tolist())

    print("\nCleaned dataset saved:", OUTPUT_PATH)


if __name__ == "__main__":
    clean_dataset()

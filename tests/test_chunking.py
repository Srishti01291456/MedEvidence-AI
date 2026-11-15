
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CLEAN_PATH = BASE_DIR / "data" / "processed" / "medquad_clean.csv"
CHUNK_PATH = BASE_DIR / "data" / "processed" / "medquad_chunks.csv"


def test_chunking():

    print("Loading datasets...")

    original = pd.read_csv(CLEAN_PATH)
    chunks = pd.read_csv(CHUNK_PATH)

    # Test 1: Chunks should exist
    assert len(chunks) > 0, "No chunks found!"

    # Test 2: Required columns
    required_columns = [
        "chunk_id",
        "original_row_index",
        "document_id",
        "question_id",
        "question",
        "answer_chunk",
        "retrieval_text",
        "source_name",
        "source_url"
    ]

    for column in required_columns:
        assert column in chunks.columns, f"Missing column: {column}"

    # Test 3: Unique chunk IDs
    assert chunks["chunk_id"].notna().all()
    assert chunks["chunk_id"].is_unique, "Duplicate chunk IDs found!"

    # Test 4: Non-empty answer chunks
    assert chunks["answer_chunk"].notna().all()

    assert (
        chunks["answer_chunk"].astype(str).str.strip() != ""
    ).all(), "Empty answer chunks found!"

    # Test 5: Every chunk refers to a valid original row
    assert chunks["original_row_index"].notna().all()

    for _, chunk in chunks.iterrows():

        row_index = int(chunk["original_row_index"])

        assert 0 <= row_index < len(original), (
            f"Invalid original row index: {row_index}"
        )

        source = original.iloc[row_index]

        # Compare original question
        assert str(chunk["question"]) == str(source["question"]), (
            "Question mismatch!"
        )

        # Compare source name, handling missing values
        original_name = (
            "" if pd.isna(source["source_name"])
            else str(source["source_name"]).strip()
        )

        chunk_name = (
            "" if pd.isna(chunk["source_name"])
            else str(chunk["source_name"]).strip()
        )

        assert chunk_name == original_name, (
            f"Source name mismatch at original row {row_index}!"
        )

        # Compare source URL, handling missing values
        original_url = (
            "" if pd.isna(source["source_url"])
            else str(source["source_url"]).strip()
        )

        chunk_url = (
            "" if pd.isna(chunk["source_url"])
            else str(chunk["source_url"]).strip()
        )

        assert chunk_url == original_url, (
            f"Source URL mismatch at original row {row_index}!"
        )

        # Check retrieval text
        assert str(chunk["question"]) in str(chunk["retrieval_text"])

        assert str(chunk["answer_chunk"]) in str(chunk["retrieval_text"])

    print("\nAll chunking and metadata tests passed!")
    print("Original records:", len(original))
    print("Total chunks:", len(chunks))
    print("Unique chunk IDs:", chunks["chunk_id"].nunique())


if __name__ == "__main__":
    test_chunking()

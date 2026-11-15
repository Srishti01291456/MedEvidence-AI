
import sys
import pandas as pd

from pathlib import Path

# Add project root to Python import path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.dense_retrieval import (
    load_embedding_model,
    load_dense_index,
    search_dense
)


DATA_PATH = BASE_DIR / "data" / "processed" / "medquad_chunks.csv"


def test_dense_retrieval():

    print("Loading chunked dataset...")

    chunks_df = pd.read_csv(DATA_PATH)

    assert len(chunks_df) > 0, "No chunks found!"

    # Load embedding model
    model = load_embedding_model()

    # Load previously created FAISS index
    index = load_dense_index()

    # Test 1: Check index size
    assert index.ntotal == len(chunks_df), (
        "FAISS index size does not match the dataset!"
    )

    # Test query
    query = "What are the symptoms of diabetes?"

    # Test 2: Perform dense retrieval
    results = search_dense(
        query,
        chunks_df,
        model,
        index,
        top_k=5
    )

    # Test 3: Check returned results
    assert len(results) > 0, "No search results found!"

    assert len(results) <= 5

    # Test 4: Check required metadata
    required_columns = [
        "chunk_id",
        "question",
        "answer_chunk",
        "source_name",
        "source_url",
        "retrieval_score"
    ]

    for column in required_columns:
        assert column in results.columns, (
            f"Missing result column: {column}"
        )

    assert results["chunk_id"].notna().all()

    assert results["retrieval_score"].notna().all()

    # Test 5: Check descending similarity scores
    scores = results["retrieval_score"].tolist()

    assert scores == sorted(scores, reverse=True), (
        "Results are not sorted by similarity score!"
    )

    # Test 6: Check empty query handling
    empty_results = search_dense(
        "",
        chunks_df,
        model,
        index,
        top_k=5
    )

    assert empty_results.empty

    # Display results
    print("\n========== DENSE RETRIEVAL TEST RESULTS ==========")

    print("Total indexed chunks:", index.ntotal)
    print("Retrieved chunks:", len(results))

    print("\nTop retrieved result:")
    print("Question:", results.iloc[0]["question"])
    print("Similarity score:", results.iloc[0]["retrieval_score"])
    print("Source:", results.iloc[0]["source_name"])
    print("URL:", results.iloc[0]["source_url"])

    print("\nAll dense retrieval tests passed!")


if __name__ == "__main__":
    test_dense_retrieval()

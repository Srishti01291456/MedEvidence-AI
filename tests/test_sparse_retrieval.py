
import sys
import pandas as pd

from pathlib import Path

# Add project root to Python import path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.sparse_retrieval import (
    build_tfidf_index,
    build_bm25_index,
    search_tfidf,
    search_bm25
)

# Dataset path
DATA_PATH = BASE_DIR / "data" / "processed" / "medquad_chunks.csv"


def test_sparse_retrieval():

    print("Loading chunked dataset...")

    chunks_df = pd.read_csv(DATA_PATH)

    # Test 1: Dataset should contain chunks
    assert len(chunks_df) > 0, "No chunks found!"

    # Test 2: Build TF-IDF index
    vectorizer, tfidf_matrix = build_tfidf_index(chunks_df)

    assert tfidf_matrix.shape[0] == len(chunks_df)
    assert tfidf_matrix.shape[1] > 0

    # Test 3: Build BM25 index
    bm25 = build_bm25_index(chunks_df)

    assert len(bm25.doc_len) == len(chunks_df)

    # Test query
    query = "What are the symptoms of diabetes?"

    # Test 4: TF-IDF search
    tfidf_results = search_tfidf(
        query,
        chunks_df,
        vectorizer,
        tfidf_matrix,
        top_k=5
    )

    assert len(tfidf_results) > 0
    assert len(tfidf_results) <= 5

    # Test 5: BM25 search
    bm25_results = search_bm25(
        query,
        chunks_df,
        bm25,
        top_k=5
    )

    assert len(bm25_results) > 0
    assert len(bm25_results) <= 5

    # Test 6: Check source metadata
    for results in [tfidf_results, bm25_results]:

        assert "source_name" in results.columns
        assert "source_url" in results.columns
        assert "chunk_id" in results.columns

        assert results["chunk_id"].notna().all()
        assert results["retrieval_score"].notna().all()

    print("\n========== TEST RESULTS ==========")

    print("TF-IDF retrieved chunks:", len(tfidf_results))
    print("BM25 retrieved chunks:", len(bm25_results))

    print("\nTF-IDF top result:")
    print(tfidf_results.iloc[0]["question"])
    print(tfidf_results.iloc[0]["source_name"])

    print("\nBM25 top result:")
    print(bm25_results.iloc[0]["question"])
    print(bm25_results.iloc[0]["source_name"])

    print("\nAll sparse retrieval tests passed!")


if __name__ == "__main__":
    test_sparse_retrieval()

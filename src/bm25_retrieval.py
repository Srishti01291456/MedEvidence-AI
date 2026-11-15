
import pandas as pd
import numpy as np
import joblib
import re

from pathlib import Path
from rank_bm25 import BM25Okapi


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "medquad_chunks.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

BM25_PATH = MODEL_DIR / "bm25_index.joblib"


# Text preprocessing
def tokenize(text):
    text = str(text).lower()
    return re.findall(r"\b[a-z0-9]+\b", text)


# Build BM25 index
def build_bm25_index(chunks_df):

    print("\nBuilding BM25 index...")

    texts = chunks_df["retrieval_text"].fillna("").astype(str)

    tokenized_corpus = [
        tokenize(text) for text in texts
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    print("BM25 index created successfully!")
    print("Total indexed chunks:", len(tokenized_corpus))

    return bm25


# Search using BM25
def search_bm25(query, bm25, chunks_df, top_k=5):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:

        result = chunks_df.iloc[idx].to_dict()
        result["bm25_score"] = float(scores[idx])

        results.append(result)

    return results


# Main execution
def main():

    print("Loading chunked MedQuAD dataset...")

    chunks_df = pd.read_csv(DATA_PATH)

    print("Total chunks:", len(chunks_df))

    # Build index
    bm25 = build_bm25_index(chunks_df)

    # Save index
    print("\nSaving BM25 index...")

    joblib.dump(bm25, BM25_PATH)

    print("BM25 index saved successfully!")

    # Test query
    query = "What are the symptoms of diabetes?"

    print("\n================================")
    print("USER QUERY:")
    print(query)
    print("================================")

    results = search_bm25(
        query,
        bm25,
        chunks_df,
        top_k=5
    )

    print("\n========== BM25 RETRIEVAL RESULTS ==========")

    for i, result in enumerate(results, start=1):

        print(f"\nResult: {i}")
        print("Chunk ID:", result["chunk_id"])
        print("BM25 Score:", round(result["bm25_score"], 4))
        print("Question:", result["question"])
        print("Answer:", result["answer_chunk"][:400])
        print("Source:", result["source_name"])
        print("URL:", result["source_url"])


if __name__ == "__main__":
    main()

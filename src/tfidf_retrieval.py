
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "medquad_chunks.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MATRIX_PATH = MODEL_DIR / "tfidf_matrix.joblib"


# Build TF-IDF index
def build_tfidf_index(chunks_df):

    print("\nBuilding TF-IDF index...")

    texts = chunks_df["retrieval_text"].fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=100000
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    print("TF-IDF index created successfully!")
    print("Matrix shape:", tfidf_matrix.shape)

    return vectorizer, tfidf_matrix


# Search using TF-IDF
def search_tfidf(query, vectorizer, tfidf_matrix, chunks_df, top_k=5):

    query_vector = vectorizer.transform([query])

    scores = (tfidf_matrix @ query_vector.T).toarray().flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:

        result = chunks_df.iloc[idx].to_dict()
        result["tfidf_score"] = float(scores[idx])

        results.append(result)

    return results


# Main execution
def main():

    print("Loading chunked MedQuAD dataset...")

    chunks_df = pd.read_csv(DATA_PATH)

    print("Total chunks:", len(chunks_df))

    # Build index
    vectorizer, tfidf_matrix = build_tfidf_index(chunks_df)

    # Save index
    print("\nSaving TF-IDF index...")

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(tfidf_matrix, MATRIX_PATH)

    print("TF-IDF index saved successfully!")

    # Test query
    query = "What are the symptoms of diabetes?"

    print("\n================================")
    print("USER QUERY:")
    print(query)
    print("================================")

    results = search_tfidf(
        query,
        vectorizer,
        tfidf_matrix,
        chunks_df,
        top_k=5
    )

    print("\n========== TF-IDF RETRIEVAL RESULTS ==========")

    for i, result in enumerate(results, start=1):

        print(f"\nResult: {i}")
        print("Chunk ID:", result["chunk_id"])
        print("TF-IDF Score:", round(result["tfidf_score"], 4))
        print("Question:", result["question"])
        print("Answer:", result["answer_chunk"][:400])
        print("Source:", result["source_name"])
        print("URL:", result["source_url"])


if __name__ == "__main__":
    main()

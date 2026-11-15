import re
import joblib
import pandas as pd

from pathlib import Path
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz, load_npz


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "medquad_chunks.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TFIDF_VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
TFIDF_MATRIX_PATH = MODEL_DIR / "tfidf_matrix.npz"
BM25_MODEL_PATH = MODEL_DIR / "bm25_model.joblib"


# --------------------------------------------------
# TEXT PREPROCESSING FOR BM25
# --------------------------------------------------

def tokenize_text(text):
    """
    Convert text into lowercase word tokens.
    """

    text = str(text).lower()

    tokens = re.findall(r"\b[a-z0-9]+\b", text)

    return tokens


# --------------------------------------------------
# BUILD TF-IDF INDEX
# --------------------------------------------------

def build_tfidf_index(chunks_df):

    print("\nBuilding TF-IDF index...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=100000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(
        chunks_df["retrieval_text"].fillna("").astype(str)
    )

    print("TF-IDF index created!")
    print("Matrix shape:", tfidf_matrix.shape)

    return vectorizer, tfidf_matrix


# --------------------------------------------------
# BUILD BM25 INDEX
# --------------------------------------------------

def build_bm25_index(chunks_df):

    print("\nBuilding BM25 index...")

    tokenized_documents = [
        tokenize_text(text)
        for text in chunks_df["retrieval_text"].fillna("")
    ]

    bm25 = BM25Okapi(tokenized_documents)

    print("BM25 index created!")
    print("Total indexed documents:", len(tokenized_documents))

    return bm25


# --------------------------------------------------
# TF-IDF SEARCH
# --------------------------------------------------

def search_tfidf(
    query,
    chunks_df,
    vectorizer,
    tfidf_matrix,
    top_k=5
):

    query_vector = vectorizer.transform([query])

    scores = (tfidf_matrix @ query_vector.T).toarray().flatten()

    top_indices = scores.argsort()[::-1][:top_k]

    results = chunks_df.iloc[top_indices].copy()

    results["retrieval_score"] = scores[top_indices]

    results = results[results["retrieval_score"] > 0]

    return results.sort_values(
        by="retrieval_score",
        ascending=False
    )


# --------------------------------------------------
# BM25 SEARCH
# --------------------------------------------------

def search_bm25(
    query,
    chunks_df,
    bm25,
    top_k=5
):

    query_tokens = tokenize_text(query)

    if not query_tokens:
        return pd.DataFrame()

    scores = bm25.get_scores(query_tokens)

    top_indices = scores.argsort()[::-1][:top_k]

    results = chunks_df.iloc[top_indices].copy()

    results["retrieval_score"] = scores[top_indices]

    results = results[results["retrieval_score"] > 0]

    return results.sort_values(
        by="retrieval_score",
        ascending=False
    )


# --------------------------------------------------
# SAVE RETRIEVAL MODELS
# --------------------------------------------------

def save_models(vectorizer, tfidf_matrix, bm25):

    print("\nSaving retrieval models...")

    joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)

    save_npz(TFIDF_MATRIX_PATH, tfidf_matrix)

    joblib.dump(bm25, BM25_MODEL_PATH)

    print("Models saved successfully!")


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():

    print("Loading chunked MedQuAD dataset...")

    chunks_df = pd.read_csv(DATA_PATH)

    print("Total chunks:", len(chunks_df))

    # Build both retrieval indexes
    vectorizer, tfidf_matrix = build_tfidf_index(chunks_df)

    bm25 = build_bm25_index(chunks_df)

    # Save indexes
    save_models(vectorizer, tfidf_matrix, bm25)

    # Sample medical query
    query = "What are the symptoms of diabetes?"

    print("\n================================")
    print("USER QUERY:")
    print(query)
    print("================================")

    # TF-IDF results
    print("\n========== TF-IDF RESULTS ==========")

    tfidf_results = search_tfidf(
        query,
        chunks_df,
        vectorizer,
        tfidf_matrix,
        top_k=5
    )

    if tfidf_results.empty:
        print("No matching results found.")
    else:
        for _, result in tfidf_results.iterrows():

            print("\nChunk ID:", result["chunk_id"])
            print("Score:", round(result["retrieval_score"], 4))
            print("Question:", result["question"])
            print("Answer:", result["answer_chunk"][:400])
            print("Source:", result["source_name"])
            print("URL:", result["source_url"])

    # BM25 results
    print("\n========== BM25 RESULTS ==========")

    bm25_results = search_bm25(
        query,
        chunks_df,
        bm25,
        top_k=5
    )

    if bm25_results.empty:
        print("No matching results found.")
    else:
        for _, result in bm25_results.iterrows():

            print("\nChunk ID:", result["chunk_id"])
            print("Score:", round(result["retrieval_score"], 4))
            print("Question:", result["question"])
            print("Answer:", result["answer_chunk"][:400])
            print("Source:", result["source_name"])
            print("URL:", result["source_url"])


if __name__ == "__main__":
    main()

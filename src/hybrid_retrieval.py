
import pandas as pd
import numpy as np
import joblib
import faiss

from pathlib import Path

from src.tfidf_retrieval import search_tfidf
from src.bm25_retrieval import search_bm25
from src.dense_retrieval import load_embedding_model


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "medquad_chunks.csv"

MODEL_DIR = BASE_DIR / "models"

TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MATRIX_PATH = MODEL_DIR / "tfidf_matrix.joblib"
BM25_PATH = MODEL_DIR / "bm25_index.joblib"
FAISS_PATH = MODEL_DIR / "medquad_faiss.index"


# ==========================================
# HYBRID RETRIEVAL WEIGHTS
# ==========================================

TFIDF_WEIGHT = 0.20
BM25_WEIGHT = 0.30
DENSE_WEIGHT = 0.50

TOP_K = 20
FINAL_TOP_K = 5


# ==========================================
# SCORE NORMALIZATION
# ==========================================

def normalize_scores(scores):

    scores = np.asarray(scores, dtype=np.float32)

    min_score = np.min(scores)
    max_score = np.max(scores)

    if max_score == min_score:
        return np.zeros(len(scores), dtype=np.float32)

    return (scores - min_score) / (max_score - min_score)


# ==========================================
# HYBRID RETRIEVAL
# ==========================================

def hybrid_search(
    query,
    chunks_df,
    vectorizer,
    tfidf_matrix,
    bm25,
    faiss_index,
    embedding_model,
    top_k=TOP_K
):

    print("\nRunning TF-IDF retrieval...")

    tfidf_results = search_tfidf(
        query,
        vectorizer,
        tfidf_matrix,
        chunks_df,
        top_k=top_k
    )

    print("Running BM25 retrieval...")

    bm25_results = search_bm25(
        query,
        bm25,
        chunks_df,
        top_k=top_k
    )

    print("Running Dense retrieval...")

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    dense_scores, dense_indices = faiss_index.search(
        query_embedding,
        top_k
    )

    # Store combined scores using chunk IDs
    score_map = {}

    # --------------------------------------
    # TF-IDF SCORES
    # --------------------------------------

    tfidf_raw = [
        result["tfidf_score"]
        for result in tfidf_results
    ]

    tfidf_normalized = normalize_scores(tfidf_raw)

    for result, score in zip(
        tfidf_results,
        tfidf_normalized
    ):

        chunk_id = result["chunk_id"]

        score_map[chunk_id] = score_map.get(
            chunk_id, 0.0
        ) + TFIDF_WEIGHT * float(score)

    # --------------------------------------
    # BM25 SCORES
    # --------------------------------------

    bm25_raw = [
        result["bm25_score"]
        for result in bm25_results
    ]

    bm25_normalized = normalize_scores(bm25_raw)

    for result, score in zip(
        bm25_results,
        bm25_normalized
    ):

        chunk_id = result["chunk_id"]

        score_map[chunk_id] = score_map.get(
            chunk_id, 0.0
        ) + BM25_WEIGHT * float(score)

    # --------------------------------------
    # DENSE RETRIEVAL SCORES
    # --------------------------------------

    dense_normalized = normalize_scores(
        dense_scores[0]
    )

    for score, idx in zip(
        dense_normalized,
        dense_indices[0]
    ):

        if idx == -1:
            continue

        chunk_id = chunks_df.iloc[int(idx)]["chunk_id"]

        score_map[chunk_id] = score_map.get(
            chunk_id, 0.0
        ) + DENSE_WEIGHT * float(score)

    # --------------------------------------
    # FINAL RESULT RANKING
    # --------------------------------------

    chunk_lookup = chunks_df.set_index("chunk_id")

    final_results = []

    for chunk_id, score in score_map.items():

        result = chunk_lookup.loc[chunk_id].to_dict()

        # FIX: Restore chunk_id after setting it as index
        result["chunk_id"] = chunk_id

        result["hybrid_score"] = float(score)

        final_results.append(result)

    final_results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    return final_results[:FINAL_TOP_K]


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():

    print("\n====================================")
    print("       MEDEVIDENCE-AI")
    print("       HYBRID RETRIEVAL")
    print("====================================")

    # Load dataset
    print("\nLoading chunked dataset...")

    chunks_df = pd.read_csv(DATA_PATH)

    print("Total chunks:", len(chunks_df))

    # Load TF-IDF
    print("\nLoading TF-IDF index...")

    vectorizer = joblib.load(TFIDF_PATH)
    tfidf_matrix = joblib.load(MATRIX_PATH)

    print("TF-IDF loaded successfully!")

    # Load BM25
    print("\nLoading BM25 index...")

    bm25 = joblib.load(BM25_PATH)

    print("BM25 loaded successfully!")

    # Load FAISS
    print("\nLoading FAISS index...")

    faiss_index = faiss.read_index(str(FAISS_PATH))

    print("FAISS loaded successfully!")

    # Load embedding model
    print("\nLoading embedding model...")

    embedding_model = load_embedding_model()

    print("Embedding model ready!")

    # User query
    query = "What are the symptoms of diabetes?"

    print("\n====================================")
    print("USER QUERY:")
    print(query)
    print("====================================")

    # Run hybrid retrieval
    results = hybrid_search(
        query,
        chunks_df,
        vectorizer,
        tfidf_matrix,
        bm25,
        faiss_index,
        embedding_model
    )

    # Display results
    print("\n========== HYBRID RETRIEVAL RESULTS ==========")

    for i, result in enumerate(results, start=1):

        print(f"\nResult: {i}")

        print("Chunk ID:", result["chunk_id"])

        print(
            "Hybrid Score:",
            round(result["hybrid_score"], 4)
        )

        print("Question:", result["question"])

        print(
            "Answer:",
            str(result["answer_chunk"])[:400]
        )

        print("Source:", result["source_name"])

        print("URL:", result["source_url"])

        print("----------------------------------------")

    print("\nHybrid retrieval completed successfully!")


if __name__ == "__main__":
    main()

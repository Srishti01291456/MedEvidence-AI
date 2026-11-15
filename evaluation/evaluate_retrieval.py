
import pandas as pd
import numpy as np
import joblib
import faiss

from pathlib import Path

from src.tfidf_retrieval import search_tfidf
from src.bm25_retrieval import search_bm25
from src.dense_retrieval import (
    load_embedding_model,
    load_dense_index,
    search_dense
)
from src.hybrid_retrieval import hybrid_search


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "medquad_chunks.csv"

EVALUATION_DIR = BASE_DIR / "evaluation"

TEST_QUERIES_PATH = EVALUATION_DIR / "test_queries.csv"

MODEL_DIR = BASE_DIR / "models"

TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MATRIX_PATH = MODEL_DIR / "tfidf_matrix.joblib"
BM25_PATH = MODEL_DIR / "bm25_index.joblib"
FAISS_PATH = MODEL_DIR / "medquad_faiss.index"

TOP_K = 5


# ==========================================
# EVALUATION METRICS
# ==========================================

def calculate_metrics(retrieved_ids, relevant_ids):

    retrieved_ids = retrieved_ids[:TOP_K]

    relevant_ids = set(relevant_ids)

    retrieved_relevant = [
        chunk_id
        for chunk_id in retrieved_ids
        if chunk_id in relevant_ids
    ]

    # Precision@5
    precision = len(retrieved_relevant) / TOP_K

    # Recall@5
    recall = (
        len(retrieved_relevant) / len(relevant_ids)
        if relevant_ids else 0.0
    )

    # Mean Reciprocal Rank
    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(retrieved_ids, start=1):

        if chunk_id in relevant_ids:
            reciprocal_rank = 1 / rank
            break

    # Hit Rate@5
    hit_rate = 1.0 if retrieved_relevant else 0.0

    return {
        "precision@5": precision,
        "recall@5": recall,
        "mrr": reciprocal_rank,
        "hit_rate@5": hit_rate
    }


# ==========================================
# EXTRACT CHUNK IDS
# ==========================================

def extract_ids(results):

    if isinstance(results, pd.DataFrame):

        return results["chunk_id"].astype(str).tolist()

    return [
        str(result["chunk_id"])
        for result in results
    ]


# ==========================================
# LOAD ALL MODELS
# ==========================================

def load_all_models():

    print("\n====================================")
    print("LOADING DATA AND RETRIEVAL MODELS")
    print("====================================")

    # Load dataset
    print("\nLoading MedQuAD dataset...")

    chunks_df = pd.read_csv(DATA_PATH)

    print("Total chunks:", len(chunks_df))

    # Load test queries
    print("\nLoading evaluation queries...")

    test_df = pd.read_csv(TEST_QUERIES_PATH)

    print("Total test queries:", len(test_df))

    # Load TF-IDF
    print("\nLoading TF-IDF models...")

    vectorizer = joblib.load(TFIDF_PATH)
    tfidf_matrix = joblib.load(MATRIX_PATH)

    print("TF-IDF loaded successfully!")

    # Load BM25
    print("\nLoading BM25 model...")

    bm25 = joblib.load(BM25_PATH)

    print("BM25 loaded successfully!")

    # Load FAISS
    print("\nLoading FAISS index...")

    faiss_index = load_dense_index()

    if faiss_index.ntotal != len(chunks_df):
        raise ValueError(
            "FAISS index size does not match dataset size. "
            "Rebuild the FAISS index."
        )

    print("FAISS loaded successfully!")

    # Load embedding model
    embedding_model = load_embedding_model()

    print("\nAll models loaded successfully!")

    return (
        chunks_df,
        test_df,
        vectorizer,
        tfidf_matrix,
        bm25,
        faiss_index,
        embedding_model
    )


# ==========================================
# RUN EVALUATION
# ==========================================

def evaluate_all_methods():

    (
        chunks_df,
        test_df,
        vectorizer,
        tfidf_matrix,
        bm25,
        faiss_index,
        embedding_model
    ) = load_all_models()

    all_results = []

    print("\n====================================")
    print("STARTING RETRIEVAL EVALUATION")
    print("====================================")

    for query_number, row in test_df.iterrows():

        query = str(row["query"])

        relevant_ids = [
            chunk_id.strip()
            for chunk_id in str(row["relevant_chunk_ids"]).split(";")
            if chunk_id.strip() and chunk_id.strip().lower() != "nan"
        ]

        print("\n------------------------------------")
        print(f"QUERY {query_number + 1}: {query}")
        print("------------------------------------")

        # ----------------------------------
        # TF-IDF
        # ----------------------------------

        tfidf_results = search_tfidf(
            query,
            vectorizer,
            tfidf_matrix,
            chunks_df,
            top_k=TOP_K
        )

        tfidf_ids = extract_ids(tfidf_results)

        # ----------------------------------
        # BM25
        # ----------------------------------

        bm25_results = search_bm25(
            query,
            bm25,
            chunks_df,
            top_k=TOP_K
        )

        bm25_ids = extract_ids(bm25_results)

        # ----------------------------------
        # DENSE
        # ----------------------------------

        dense_results = search_dense(
            query,
            chunks_df,
            embedding_model,
            faiss_index,
            top_k=TOP_K
        )

        dense_ids = extract_ids(dense_results)

        # ----------------------------------
        # HYBRID
        # ----------------------------------

        hybrid_results = hybrid_search(
            query,
            chunks_df,
            vectorizer,
            tfidf_matrix,
            bm25,
            faiss_index,
            embedding_model,
            top_k=20
        )

        hybrid_ids = extract_ids(hybrid_results)

        # ----------------------------------
        # CALCULATE METRICS
        # ----------------------------------

        methods = {
            "TF-IDF": tfidf_ids,
            "BM25": bm25_ids,
            "Dense": dense_ids,
            "Hybrid": hybrid_ids
        }

        for method_name, retrieved_ids in methods.items():

            metrics = calculate_metrics(
                retrieved_ids,
                relevant_ids
            )

            result = {
                "query": query,
                "method": method_name,
                "relevant_chunks": len(relevant_ids),
                "retrieved_chunk_ids": ";".join(retrieved_ids),
                **metrics
            }

            all_results.append(result)

            print(
                f"{method_name:10} | "
                f"Precision: {metrics['precision@5']:.2f} | "
                f"Recall: {metrics['recall@5']:.2f} | "
                f"MRR: {metrics['mrr']:.2f} | "
                f"Hit Rate: {metrics['hit_rate@5']:.2f}"
            )

    # ======================================
    # SAVE QUERY-WISE RESULTS
    # ======================================

    results_df = pd.DataFrame(all_results)

    results_path = (
        EVALUATION_DIR / "retrieval_evaluation_results.csv"
    )

    results_df.to_csv(results_path, index=False)

    # ======================================
    # CALCULATE AVERAGE METRICS
    # ======================================

    summary_df = (
        results_df
        .groupby("method")[
            ["precision@5", "recall@5", "mrr", "hit_rate@5"]
        ]
        .mean()
        .reset_index()
    )

    summary_df = summary_df.round(4)

    summary_path = (
        EVALUATION_DIR / "retrieval_evaluation_summary.csv"
    )

    summary_df.to_csv(summary_path, index=False)

    # ======================================
    # DISPLAY FINAL COMPARISON
    # ======================================

    print("\n\n====================================")
    print("FINAL RETRIEVAL EVALUATION")
    print("====================================")

    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\nEvaluation completed successfully!")

    print("\nSaved files:")
    print(results_path)
    print(summary_path)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    evaluate_all_methods()

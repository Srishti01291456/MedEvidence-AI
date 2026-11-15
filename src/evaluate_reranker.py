
import pandas as pd

from src.dense_retrieval import (
    DATA_PATH,
    load_embedding_model,
    load_dense_index,
    search_dense
)

from src.reranker import (
    load_reranker,
    rerank_chunks
)


TEST_PATH = "evaluation/test_queries.csv"

CANDIDATE_TOP_K = 10
FINAL_TOP_K = 5


# -----------------------------------------
# 1. Calculate evaluation metrics
# -----------------------------------------

def calculate_metrics(results, relevant_ids):

    relevant_ids = set(relevant_ids)

    retrieved_ids = [
        str(chunk_id)
        for chunk_id in results["chunk_id"].tolist()
    ]

    hits = [
        chunk_id
        for chunk_id in retrieved_ids
        if chunk_id in relevant_ids
    ]

    precision = len(hits) / FINAL_TOP_K

    recall = (
        len(hits) / len(relevant_ids)
        if relevant_ids else 0
    )

    reciprocal_rank = 0

    for rank, chunk_id in enumerate(retrieved_ids, start=1):

        if chunk_id in relevant_ids:
            reciprocal_rank = 1 / rank
            break

    hit_rate = 1 if len(hits) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "mrr": reciprocal_rank,
        "hit_rate": hit_rate
    }


# -----------------------------------------
# 2. Load models and dataset
# -----------------------------------------

def evaluate_reranker():

    print("\n===================================")
    print("RERANKER EVALUATION")
    print("===================================")

    chunks_df = pd.read_csv(DATA_PATH)

    test_df = pd.read_csv(TEST_PATH)

    print("Total chunks:", len(chunks_df))
    print("Test queries:", len(test_df))

    print("\nLoading dense retrieval model...")

    embedding_model = load_embedding_model()

    faiss_index = load_dense_index()

    print("\nLoading MedCPT reranker...")

    tokenizer, reranker_model, device = load_reranker()

    dense_metrics = []
    reranker_metrics = []
    detailed_results = []

    # -----------------------------------------
    # 3. Evaluate every test query
    # -----------------------------------------

    for index, row in test_df.iterrows():

        query = str(row["query"])

        relevant_ids = [
            item.strip()
            for item in str(row["relevant_chunk_ids"]).split(";")
            if item.strip()
        ]

        print("\n-----------------------------------")
        print("Query:", query)

        # Retrieve 10 candidate chunks
        candidates = search_dense(
            query=query,
            chunks_df=chunks_df,
            model=embedding_model,
            index=faiss_index,
            top_k=CANDIDATE_TOP_K
        )

        # Dense retrieval top 5
        dense_top5 = candidates.head(FINAL_TOP_K).copy()

        # Rerank all 10 candidates
        reranked = rerank_chunks(
            query=query,
            chunks_df=candidates,
            tokenizer=tokenizer,
            model=reranker_model,
            device=device
        )

        # Select reranked top 5
        reranked_top5 = reranked.head(FINAL_TOP_K).copy()

        # Calculate metrics
        dense_score = calculate_metrics(
            dense_top5,
            relevant_ids
        )

        reranker_score = calculate_metrics(
            reranked_top5,
            relevant_ids
        )

        # Candidate Recall@10
        candidate_ids = set(
            candidates["chunk_id"].astype(str)
        )

        candidate_hits = candidate_ids.intersection(
            set(relevant_ids)
        )

        candidate_recall = (
            len(candidate_hits) / len(relevant_ids)
            if relevant_ids else 0
        )

        dense_metrics.append(dense_score)

        reranker_metrics.append(reranker_score)

        detailed_results.append({
            "query": query,
            "candidate_recall@10": candidate_recall,
            "dense_precision@5": dense_score["precision"],
            "reranker_precision@5": reranker_score["precision"],
            "dense_recall@5": dense_score["recall"],
            "reranker_recall@5": reranker_score["recall"],
            "dense_mrr": dense_score["mrr"],
            "reranker_mrr": reranker_score["mrr"],
            "dense_hit_rate@5": dense_score["hit_rate"],
            "reranker_hit_rate@5": reranker_score["hit_rate"]
        })

        print("\nDense Top 5:")

        print(
            dense_top5["chunk_id"].tolist()
        )

        print("\nReranked Top 5:")

        print(
            reranked_top5[
                ["chunk_id", "rerank_score"]
            ].to_string(index=False)
        )

    # -----------------------------------------
    # 4. Calculate average metrics
    # -----------------------------------------

    dense_average = pd.DataFrame(
        dense_metrics
    ).mean()

    reranker_average = pd.DataFrame(
        reranker_metrics
    ).mean()

    comparison = pd.DataFrame({

        "Dense Retrieval": dense_average,

        "Dense + MedCPT Reranking": reranker_average

    })

    # Add candidate recall
    candidate_recall_average = pd.DataFrame(
        detailed_results
    )["candidate_recall@10"].mean()

    print("\n===================================")
    print("FINAL EVALUATION RESULTS")
    print("===================================")

    print(
        comparison.round(4).to_string()
    )

    print(
        "\nDense Candidate Recall@10:",
        round(candidate_recall_average, 4)
    )

    # Save detailed results
    results_df = pd.DataFrame(detailed_results)

    results_df.to_csv(
        "evaluation/reranker_evaluation_results.csv",
        index=False
    )

    print(
        "\nDetailed results saved to:"
        "\nevaluation/reranker_evaluation_results.csv"
    )


if __name__ == "__main__":

    evaluate_reranker()

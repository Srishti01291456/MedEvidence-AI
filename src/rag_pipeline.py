
import pandas as pd

from src.dense_retrieval import (
    DATA_PATH,
    load_embedding_model,
    load_dense_index,
    search_dense
)

from src.generation import generate_answer


# Configuration
MIN_RELEVANCE_SCORE = 0.40
FINAL_TOP_K = 5


# -----------------------------------------
# 1. Initialize RAG system
# -----------------------------------------

def initialize_rag_system():

    print("\n===================================")
    print("INITIALIZING MEDEVIDENCE-AI RAG")
    print("===================================")

    print("\nLoading MedQuAD dataset...")

    chunks_df = pd.read_csv(DATA_PATH)

    print("Total medical chunks:", len(chunks_df))

    print("\nLoading dense embedding model...")
    model = load_embedding_model()

    print("\nLoading FAISS index...")
    index = load_dense_index()

    if index.ntotal != len(chunks_df):
        raise ValueError(
            "FAISS index and dataset size do not match. "
            "Please rebuild the index."
        )

    print("\nRAG system initialized successfully!")

    return {
        "chunks_df": chunks_df,
        "model": model,
        "index": index
    }


# -----------------------------------------
# 2. Answer medical question
# -----------------------------------------

def answer_question(question, rag_system, top_k=10):

    if not question or not question.strip():

        return {
            "answer": "Please enter a medical question.",
            "sources": [],
            "retrieved_chunks": pd.DataFrame()
        }

    print("\n===================================")
    print("USER QUESTION:")
    print(question)
    print("===================================")

    # STEP 1: Dense Retrieval
    print("\nSTEP 1: Dense Retrieval")

    retrieved_chunks = search_dense(
        query=question,
        chunks_df=rag_system["chunks_df"],
        model=rag_system["model"],
        index=rag_system["index"],
        top_k=top_k
    )

    print("Initially retrieved chunks:", len(retrieved_chunks))

    if retrieved_chunks.empty:

        return {
            "answer": (
                "I could not find relevant medical information "
                "in the available knowledge base."
            ),
            "sources": [],
            "retrieved_chunks": retrieved_chunks
        }

    # STEP 2: Relevance Filtering
    print("\nSTEP 2: Relevance Filtering")

    highest_score = float(
        retrieved_chunks["retrieval_score"].max()
    )

    print("Highest dense score:", round(highest_score, 4))

    relevant_chunks = retrieved_chunks[
        retrieved_chunks["retrieval_score"] >= MIN_RELEVANCE_SCORE
    ].copy()

    print("Chunks passing threshold:", len(relevant_chunks))

    if relevant_chunks.empty:

        return {
            "answer": (
                "I could not find sufficiently relevant medical "
                "information in the MedQuAD knowledge base "
                "to answer this question reliably."
            ),
            "sources": [],
            "retrieved_chunks": relevant_chunks
        }

    # STEP 3: Select top evidence
    print("\nSTEP 3: Selecting Final Evidence")

    final_chunks = relevant_chunks.head(FINAL_TOP_K).copy()

    print("Final evidence chunks selected:", len(final_chunks))

    print("\nSelected evidence:")

    print(
        final_chunks[
            ["chunk_id", "retrieval_score"]
        ].to_string(index=False)
    )

    # STEP 4: Generate Answer Using Groq
    print("\nSTEP 4: Generating Answer Using Groq")

    result = generate_answer(
        user_query=question,
        retrieved_chunks=final_chunks
    )

    print("\nAnswer generated successfully!")

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "retrieved_chunks": final_chunks
    }

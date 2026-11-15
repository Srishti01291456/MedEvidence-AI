
import sys
import faiss
import numpy as np
import pandas as pd

from pathlib import Path
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "medquad_chunks.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FAISS_INDEX_PATH = MODEL_DIR / "medquad_faiss.index"

# Sentence Transformer model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

def load_embedding_model():

    print("\nLoading Sentence Transformer model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded successfully!")

    return model


# --------------------------------------------------
# BUILD DENSE INDEX
# --------------------------------------------------

def build_dense_index(chunks_df, model):

    print("\nGenerating embeddings for medical chunks...")

    texts = (
        chunks_df["retrieval_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    print("\nEmbedding generation completed!")
    print("Embedding shape:", embeddings.shape)

    # Create FAISS index using inner product.
    # Since embeddings are normalized, inner product = cosine similarity.
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print("FAISS index created successfully!")
    print("Total indexed chunks:", index.ntotal)

    return index


# --------------------------------------------------
# SAVE FAISS INDEX
# --------------------------------------------------

def save_dense_index(index):

    print("\nSaving FAISS index...")

    faiss.write_index(index, str(FAISS_INDEX_PATH))

    print("FAISS index saved at:", FAISS_INDEX_PATH)


# --------------------------------------------------
# LOAD FAISS INDEX
# --------------------------------------------------

def load_dense_index():

    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run python -m src.dense_retrieval first."
        )

    print("\nLoading saved FAISS index...")

    index = faiss.read_index(str(FAISS_INDEX_PATH))

    print("FAISS index loaded successfully!")

    return index


# --------------------------------------------------
# DENSE SEARCH
# --------------------------------------------------

def search_dense(query, chunks_df, model, index, top_k=5):

    if not query or not query.strip():
        return pd.DataFrame()

    if index.ntotal != len(chunks_df):
        raise ValueError(
            "FAISS index and chunk dataset have different sizes. "
            "Rebuild the FAISS index."
        )

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    top_k = min(top_k, index.ntotal)

    scores, indices = index.search(query_embedding, top_k)

    valid_results = [
        (int(idx), float(score))
        for idx, score in zip(indices[0], scores[0])
        if idx >= 0
    ]

    if not valid_results:
        return pd.DataFrame()

    result_indices = [idx for idx, score in valid_results]
    result_scores = [score for idx, score in valid_results]

    results = chunks_df.iloc[result_indices].copy()

    results["retrieval_score"] = result_scores

    return results.reset_index(drop=True)


# --------------------------------------------------
# DISPLAY SEARCH RESULTS
# --------------------------------------------------

def display_results(results):

    if results.empty:
        print("No results found.")
        return

    for rank, (_, result) in enumerate(results.iterrows(), start=1):

        print("\nResult:", rank)
        print("Chunk ID:", result["chunk_id"])
        print("Similarity score:", round(result["retrieval_score"], 4))
        print("Question:", result["question"])
        print("Answer:", result["answer_chunk"][:400])
        print("Source:", result["source_name"])
        print("URL:", result["source_url"])


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():

    print("Loading chunked MedQuAD dataset...")

    chunks_df = pd.read_csv(DATA_PATH)

    print("Total chunks:", len(chunks_df))

    # Load embedding model
    model = load_embedding_model()

    # Generate embeddings and create FAISS index
    index = build_dense_index(chunks_df, model)

    # Save FAISS index
    save_dense_index(index)

    # Sample medical query
    query = "What are the symptoms of diabetes?"

    print("\n================================")
    print("USER QUERY:")
    print(query)
    print("================================")

    # Retrieve relevant chunks
    results = search_dense(
        query,
        chunks_df,
        model,
        index,
        top_k=5
    )

    print("\n========== DENSE RETRIEVAL RESULTS ==========")

    display_results(results)


if __name__ == "__main__":
    main()

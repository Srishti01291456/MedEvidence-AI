
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


MODEL_NAME = "ncbi/MedCPT-Cross-Encoder"


# -----------------------------------------
# 1. Load medical reranker model
# -----------------------------------------

def load_reranker():

    print("\nLoading MedCPT Cross-Encoder...")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    print("MedCPT Cross-Encoder loaded successfully!")
    print("Reranker device:", device)

    return tokenizer, model, device


# -----------------------------------------
# 2. Rerank retrieved medical chunks
# -----------------------------------------

def rerank_chunks(
    query,
    chunks_df,
    tokenizer,
    model,
    device
):

    if chunks_df is None or chunks_df.empty:
        return chunks_df

    pairs = []

    for _, row in chunks_df.iterrows():

        question = str(row.get("question", ""))

        answer = str(row.get("answer_chunk", ""))

        medical_text = question + " " + answer

        pairs.append([query, medical_text])

    # Tokenize query-evidence pairs
    encoded = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    # Calculate relevance scores
    with torch.no_grad():

        outputs = model(**encoded)

        scores = outputs.logits.squeeze(-1)

    scores = scores.cpu().numpy()

    # Attach scores to retrieved chunks
    reranked_df = chunks_df.copy()

    reranked_df["rerank_score"] = scores

    # Sort from highest to lowest relevance
    reranked_df = reranked_df.sort_values(
        by="rerank_score",
        ascending=False
    )

    reranked_df = reranked_df.reset_index(drop=True)

    return reranked_df

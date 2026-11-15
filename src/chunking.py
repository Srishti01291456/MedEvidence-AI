
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "medquad_clean.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "medquad_chunks.csv"

CHUNK_SIZE = 180
CHUNK_OVERLAP = 30


def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    words = str(text).split()
    chunks = []

    if not words:
        return chunks

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    start = 0

    while start < len(words):

        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])

        if chunk_text.strip():
            chunks.append(chunk_text)

        if end == len(words):
            break

        start = end - overlap

    return chunks


def create_chunks():

    print("Loading cleaned dataset...")

    df = pd.read_csv(INPUT_PATH)
    all_chunks = []

    for row_index, row in df.iterrows():

        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()

        if not question or not answer:
            continue

        if question.lower() == "nan" or answer.lower() == "nan":
            continue

        answer_chunks = split_into_chunks(answer)

        for chunk_index, answer_chunk in enumerate(answer_chunks):

            chunk_id = (
                f"MEDQUAD_ROW{row_index:06d}_"
                f"CHUNK{chunk_index + 1:03d}"
            )

            chunk = {
                "chunk_id": chunk_id,

                # Exact original row reference
                "original_row_index": row_index,

                "document_id": row["document_id"],
                "question_id": row["question_id"],

                "question": question,
                "answer_chunk": answer_chunk,

                "retrieval_text": (
                    "Question: " + question
                    + "\nAnswer: " + answer_chunk
                ),

                "source_name": row["source_name"],
                "source_url": row["source_url"],
                "question_type": row["question_type"],
                "question_focus": row["question_focus"]
            }

            all_chunks.append(chunk)

    chunks_df = pd.DataFrame(all_chunks)

    if chunks_df.empty:
        raise ValueError("No chunks were generated.")

    if not chunks_df["chunk_id"].is_unique:
        raise ValueError("Duplicate chunk IDs detected!")

    chunks_df.to_csv(OUTPUT_PATH, index=False)

    print("\n========== CHUNKING REPORT ==========")
    print("Original records:", len(df))
    print("Total chunks:", len(chunks_df))
    print("Unique chunk IDs:", chunks_df["chunk_id"].nunique())
    print("Unique documents:", chunks_df["document_id"].nunique())
    print("Chunks with source names:", chunks_df["source_name"].notna().sum())
    print("Chunks with source URLs:", chunks_df["source_url"].notna().sum())

    print("\nExample chunk:")
    print(chunks_df.iloc[0]["retrieval_text"])

    print("\nExample source:")
    print(chunks_df.iloc[0]["source_name"])
    print(chunks_df.iloc[0]["source_url"])

    print("\nSaved chunks to:", OUTPUT_PATH)


if __name__ == "__main__":
    create_chunks()


import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

results_path = BASE_DIR / "evaluation" / "retrieval_evaluation_results.csv"
queries_path = BASE_DIR / "evaluation" / "test_queries.csv"
chunks_path = BASE_DIR / "data" / "processed" / "medquad_chunks.csv"

results_df = pd.read_csv(results_path)
queries_df = pd.read_csv(queries_path)
chunks_df = pd.read_csv(chunks_path)

# Create a lookup using chunk IDs
chunk_lookup = chunks_df.set_index("chunk_id")

inspection = []

for _, result in results_df.iterrows():

    query = result["query"]
    method = result["method"]

    query_row = queries_df[queries_df["query"] == query].iloc[0]

    relevant_ids = set(
        str(query_row["relevant_chunk_ids"]).split(";")
    )

    retrieved_ids = str(
        result["retrieved_chunk_ids"]
    ).split(";")

    for rank, chunk_id in enumerate(retrieved_ids, start=1):

        chunk_id = chunk_id.strip()

        if not chunk_id or chunk_id not in chunk_lookup.index:
            continue

        chunk = chunk_lookup.loc[chunk_id]

        inspection.append({
            "query": query,
            "method": method,
            "rank": rank,
            "chunk_id": chunk_id,
            "is_relevant": chunk_id in relevant_ids,
            "source_name": chunk["source_name"],
            "question": chunk["question"],
            "answer_chunk": chunk["answer_chunk"],
            "source_url": chunk["source_url"]
        })

inspection_df = pd.DataFrame(inspection)

output_path = (
    BASE_DIR / "evaluation" / "retrieval_inspection.csv"
)

inspection_df.to_csv(output_path, index=False)

print("Retrieval inspection completed!")
print("Total retrieved chunks:", len(inspection_df))
print("Saved at:", output_path)

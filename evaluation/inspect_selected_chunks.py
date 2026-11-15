
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE_DIR / "data" / "processed" / "medquad_chunks.csv"
)

selected_ids = [
    "MEDQUAD_ROW008981_CHUNK001",
    "MEDQUAD_ROW004750_CHUNK001",
    "MEDQUAD_ROW004750_CHUNK002"
]

for chunk_id in selected_ids:

    row = df[df["chunk_id"] == chunk_id]

    print("\n" + "=" * 80)

    if row.empty:
        print("Chunk not found:", chunk_id)
        continue

    row = row.iloc[0]

    print("CHUNK ID:", chunk_id)
    print("SOURCE:", row["source_name"])
    print("QUESTION:", row["question"])
    print("\nFULL ANSWER:")
    print(row["answer_chunk"])

    print("=" * 80)

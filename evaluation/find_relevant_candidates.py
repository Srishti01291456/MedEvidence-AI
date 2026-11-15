
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "evaluation" / "retrieval_inspection.csv"

df = pd.read_csv(file_path)

queries = [
    "What causes High Blood Pressure?",
    "What are the symptoms of Asthma?",
    "What are the symptoms of Basilar migraine?",
    "What are the symptoms of Permanent neonatal diabetes mellitus?"
]

for query in queries:

    print("\n" + "=" * 70)
    print("QUERY:", query)
    print("=" * 70)

    results = df[
        (df["query"] == query) &
        (df["is_relevant"] == False)
    ]

    # Remove duplicate chunks retrieved by different methods
    results = results.drop_duplicates(subset=["chunk_id"])

    for _, row in results.iterrows():

        print("\nChunk ID:", row["chunk_id"])
        print("Source:", row["source_name"])
        print("Question:", row["question"])
        print("Answer:", str(row["answer_chunk"])[:250])
        print("-" * 50)

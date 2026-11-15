
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "evaluation" / "retrieval_inspection.csv"

df = pd.read_csv(file_path)

print("\n========== RETRIEVAL SUMMARY ==========\n")

print("Total chunks:", len(df))

print("\nRelevant chunks by method:")
print(
    df.groupby("method")["is_relevant"]
    .sum()
)

print("\nRelevant percentage by method:")
print(
    (df.groupby("method")["is_relevant"].mean() * 100)
    .round(2)
)

print("\n========== TOP 5 RESULTS PER METHOD ==========\n")

for method in df["method"].unique():

    print("\nMETHOD:", method)

    method_df = df[df["method"] == method]

    for query in method_df["query"].unique():

        print("\nQUERY:", query)

        query_df = method_df[
            method_df["query"] == query
        ].sort_values("rank")

        for _, row in query_df.head(5).iterrows():

            print("\nRank:", row["rank"])
            print("Relevant:", row["is_relevant"])
            print("Source:", row["source_name"])
            print("Question:", row["question"])
            print("Answer:", str(row["answer_chunk"])[:350])
            print("-" * 50)

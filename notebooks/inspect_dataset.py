import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "data" / "raw" / "medquad_raw.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET OVERVIEW ==========")

print("\nShape:")
print(df.shape)

print("\nFirst 5 records:")
print(df.head())

print("\nColumn information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nUnique sources:")
print(df["document_source"].nunique())

print("\nRecords by source:")
print(df["document_source"].value_counts())

print("\nQuestion types:")
print(df["question_type"].value_counts().head(15))

print("\nExample question:")
print(df.iloc[0]["question"])

print("\nExample answer:")
print(df.iloc[0]["answer"])

print("\nExample source:")
print(df.iloc[0]["document_source"])

print("\nExample URL:")
print(df.iloc[0]["document_url"])

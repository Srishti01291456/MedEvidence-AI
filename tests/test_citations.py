
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed" / "medquad_clean.csv"
)

df = pd.read_csv(DATA_PATH)

assert df["question"].notna().all()
assert df["answer"].notna().all()

assert "source_name" in df.columns
assert "source_url" in df.columns

assert (
    df["source_name"].fillna("")
    .equals(df["document_source"].fillna(""))
)

assert (
    df["source_url"].fillna("")
    .equals(df["document_url"].fillna(""))
)

assert df["retrieval_text"].notna().all()

print("All dataset and source preservation tests passed!")

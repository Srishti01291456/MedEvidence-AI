
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "medquad_chunks.csv"

df = pd.read_csv(DATA_PATH)

tests = [
    ("DIABETES", ["diabetes"], ["symptom", "signs", "cause"]),
    ("HIGH BLOOD PRESSURE", ["high blood pressure"], ["symptom", "signs", "cause"]),
    ("ASTHMA", ["asthma"], ["symptom", "signs", "cause"]),
    ("ANEMIA", ["anemia"], ["symptom", "signs", "cause"]),
    ("MIGRAINE", ["migraine"], ["symptom", "signs", "cause"])
]

for disease, disease_terms, question_terms in tests:

    print("\n" + "=" * 75)
    print("DISEASE:", disease)
    print("=" * 75)

    disease_mask = df["question"].str.contains(
        "|".join(disease_terms),
        case=False,
        na=False,
        regex=True
    )

    question_mask = df["question"].str.contains(
        "|".join(question_terms),
        case=False,
        na=False,
        regex=True
    )

    matches = df[disease_mask & question_mask]

    if matches.empty:
        print("No matching questions found. Showing general disease matches.")

        matches = df[disease_mask]

    for _, row in matches.head(10).iterrows():

        print("\nChunk ID:", row["chunk_id"])
        print("Question:", row["question"])
        print("Answer:", str(row["answer_chunk"])[:400])
        print("Source:", row["source_name"])

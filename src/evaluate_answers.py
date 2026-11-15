
import csv
from pathlib import Path


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVALUATION_DIR = PROJECT_ROOT / "evaluation"

INPUT_FILE = EVALUATION_DIR / "answer_evaluation.csv"

OUTPUT_FILE = EVALUATION_DIR / "answer_evaluation_results.csv"


METRICS = [
    "relevance",
    "faithfulness",
    "completeness",
    "citation_correctness",
    "medical_safety"
]


TEST_QUESTIONS = [
    {
        "query_id": "Q1",
        "question": "What are the symptoms of sideroblastic anemia?"
    },
    {
        "query_id": "Q2",
        "question": "What causes high blood pressure?"
    },
    {
        "query_id": "Q3",
        "question": "What are the symptoms of asthma?"
    },
    {
        "query_id": "Q4",
        "question": "What are the symptoms of X-linked sideroblastic anemia?"
    },
    {
        "query_id": "Q5",
        "question": "How does a computer processor work?"
    }
]


# ==========================================
# CREATE EVALUATION TEMPLATE
# ==========================================

def create_template():

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Never overwrite an existing evaluation file
    if INPUT_FILE.exists():

        print("\nEvaluation CSV already exists.")

        print(f"File location: {INPUT_FILE}")

        print("Existing data has been preserved.")

        return

    fieldnames = [
        "query_id",
        "question",
        "answer_text",
        "relevance",
        "faithfulness",
        "completeness",
        "citation_correctness",
        "medical_safety",
        "notes"
    ]

    with open(
        INPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for item in TEST_QUESTIONS:

            writer.writerow({
                "query_id": item["query_id"],
                "question": item["question"],
                "answer_text": "",
                "relevance": "",
                "faithfulness": "",
                "completeness": "",
                "citation_correctness": "",
                "medical_safety": "",
                "notes": ""
            })

    print("\nEvaluation template created successfully!")

    print(f"\nFile location:\n{INPUT_FILE}")

    print("\nNext steps:")
    print("1. Open answer_evaluation.csv.")
    print("2. Review your chatbot answers.")
    print("3. Enter scores from 1 to 5.")
    print("4. Save the CSV file.")
    print("5. Run this program again.")


# ==========================================
# LOAD EVALUATION DATA
# ==========================================

def load_evaluation_data():

    try:

        with open(
            INPUT_FILE,
            "r",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            reader = csv.DictReader(file)

            if not reader.fieldnames:

                raise ValueError(
                    "CSV file has no header row."
                )

            required_columns = [
                "query_id",
                "question"
            ] + METRICS

            missing_columns = [
                column
                for column in required_columns
                if column not in reader.fieldnames
            ]

            if missing_columns:

                raise ValueError(
                    "Missing CSV columns: "
                    + ", ".join(missing_columns)
                )

            rows = list(reader)

        return rows

    except csv.Error as error:

        raise ValueError(
            f"Invalid CSV file: {error}"
        )


# ==========================================
# VALIDATE SCORES
# ==========================================

def validate_score(value):

    if value is None or not str(value).strip():

        return None

    try:

        score = float(value)

    except (ValueError, TypeError):

        return None

    if score < 1 or score > 5:

        return None

    if not score.is_integer():

        return None

    return int(score)


# ==========================================
# EVALUATE ANSWERS
# ==========================================

def evaluate_answers():

    print("\n======================================")
    print("      MEDEVIDENCE-AI ANSWER EVALUATION")
    print("======================================")

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Create template only if file does not exist
    if not INPUT_FILE.exists():

        print("\nEvaluation file not found.")

        create_template()

        return

    # Load CSV
    try:

        rows = load_evaluation_data()

    except ValueError as error:

        print(f"\nError: {error}")

        print("Please check your CSV file.")

        return

    # Handle empty CSV without overwriting it
    if not rows:

        print("\nNo evaluation records found.")

        print(
            "Your existing CSV has been preserved."
        )

        print(
            "\nPlease add evaluation records manually."
        )

        return

    valid_rows = []

    incomplete_rows = []

    # ======================================
    # VALIDATE EACH QUESTION
    # ======================================

    for row in rows:

        query_id = row.get(
            "query_id",
            "Unknown"
        )

        scores = {}

        for metric in METRICS:

            scores[metric] = validate_score(
                row.get(metric, "")
            )

        if all(
            score is not None
            for score in scores.values()
        ):

            valid_rows.append({
                "query_id": query_id,
                "question": row.get("question", ""),
                "scores": scores
            })

        else:

            incomplete_rows.append(query_id)

    # ======================================
    # DISPLAY EVALUATION STATUS
    # ======================================

    print(f"\nTotal questions: {len(rows)}")

    print(
        f"Completed evaluations: {len(valid_rows)}"
    )

    print(
        f"Incomplete evaluations: {len(incomplete_rows)}"
    )

    if incomplete_rows:

        print("\nQuestions needing evaluation:")

        for query_id in incomplete_rows:

            print(f"- {query_id}")

    if not valid_rows:

        print("\nNo completely evaluated questions found.")

        print(
            "Enter a valid score from 1 to 5 "
            "for all five metrics."
        )

        return

    # ======================================
    # CALCULATE METRIC AVERAGES
    # ======================================

    metric_results = {}

    print("\n======================================")
    print("             METRIC RESULTS")
    print("======================================")

    for metric in METRICS:

        scores = [
            row["scores"][metric]
            for row in valid_rows
        ]

        average = sum(scores) / len(scores)

        percentage = (average / 5) * 100

        metric_results[metric] = {
            "average": average,
            "percentage": percentage
        }

        print(
            f"\n{metric.replace('_', ' ').title()}"
        )

        print(
            f"Average score: {average:.2f}/5"
        )

        print(
            f"Percentage: {percentage:.2f}%"
        )

    # ======================================
    # CALCULATE OVERALL PERFORMANCE
    # ======================================

    overall_score = sum(
        result["average"]
        for result in metric_results.values()
    ) / len(METRICS)

    overall_percentage = (
        overall_score / 5
    ) * 100

    print("\n======================================")
    print("          OVERALL PERFORMANCE")
    print("======================================")

    print(
        f"\nOverall score: {overall_score:.2f}/5"
    )

    print(
        f"Overall percentage: {overall_percentage:.2f}%"
    )

    # ======================================
    # IDENTIFY IMPROVEMENT AREA
    # ======================================

    lowest_metric = min(
        metric_results,
        key=lambda metric: metric_results[metric]["average"]
    )

    print("\n======================================")
    print("          IMPROVEMENT AREA")
    print("======================================")

    print(
        f"\nMetric needing most attention: "
        f"{lowest_metric.replace('_', ' ').title()}"
    )

    print(
        f"Average score: "
        f"{metric_results[lowest_metric]['average']:.2f}/5"
    )

    # ======================================
    # SAVE RESULTS
    # ======================================

    fieldnames = [
        "result_type",
        "metric",
        "average_score",
        "percentage",
        "evaluated_questions"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for metric, result in metric_results.items():

            writer.writerow({
                "result_type": "Metric",
                "metric": metric,
                "average_score": round(
                    result["average"], 2
                ),
                "percentage": round(
                    result["percentage"], 2
                ),
                "evaluated_questions": len(valid_rows)
            })

        writer.writerow({
            "result_type": "Overall",
            "metric": "Overall Answer Quality",
            "average_score": round(
                overall_score, 2
            ),
            "percentage": round(
                overall_percentage, 2
            ),
            "evaluated_questions": len(valid_rows)
        })

    # ======================================
    # COMPLETION MESSAGE
    # ======================================

    print("\n======================================")
    print("          EVALUATION SAVED")
    print("======================================")

    print(
        f"\nResults saved at:\n{OUTPUT_FILE}"
    )

    print("\nAnswer evaluation completed successfully!")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    evaluate_answers()

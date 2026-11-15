
from src.rag_pipeline import (
    initialize_rag_system,
    answer_question
)


def main():

    print("\n======================================")
    print("       MEDEVIDENCE-AI CHATBOT")
    print("======================================")

    print("\nInitializing the system...")

    # Initialize RAG system only once
    rag_system = initialize_rag_system()

    print("\nChatbot is ready!")
    print("Ask a medical question.")
    print("Type 'exit' to close the chatbot.")

    while True:

        print("\n--------------------------------------")

        question = input("\nYou: ").strip()

        if question.lower() in ["exit", "quit", "stop"]:
            print("\nThank you for using MedEvidence-AI!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:

            result = answer_question(
                question=question,
                rag_system=rag_system,
                top_k=5
            )

            # Display retrieved evidence
            print("\n========== RETRIEVED EVIDENCE ==========")

            retrieved = result["retrieved_chunks"]

            for rank, (_, chunk) in enumerate(
                retrieved.iterrows(), start=1
            ):

                print(f"\nEVIDENCE [{rank}]")

                print("Chunk ID:", chunk["chunk_id"])
                print("Source:", chunk["source_name"])
                print("Question:", chunk["question"])

                print("\nMedical Information:")
                print(chunk["answer_chunk"])

                print("\nRetrieval Score:", chunk["retrieval_score"])

                print("-" * 70)

            # Display generated answer
            print("\n========== GENERATED ANSWER ==========")

            print(result["answer"])

            # Display sources
            print("\n========== SOURCES ==========")

            if result["sources"]:

                for source in result["sources"]:

                    print(
                        f"\n[{source['evidence_number']}] "
                        f"{source['source_name']}"
                    )

                    print("Chunk ID:", source["chunk_id"])

                    url = source["source_url"]

                    if url and str(url).lower() != "nan":
                        print("URL:", url)

            else:
                print("No sources available.")

            print("\n======================================")

        except Exception as e:

            print("\nAn error occurred:")
            print(type(e).__name__, e)


if __name__ == "__main__":
    main()

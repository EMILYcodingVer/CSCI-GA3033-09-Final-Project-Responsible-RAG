from answer_pipeline import run_pipeline


def pretty_print_pipeline(result) -> None:
    """Nicely print the pipeline result to the console."""
    print("===== Responsible RAG Pipeline Output =====\n")

    print("--- QUERY ---")
    print(result["query"])
    print()

    print("--- RETRIEVED_DOCS ---")
    for i, (doc, src) in enumerate(
        zip(result["retrieved_docs"], result["sources"]), start=1
    ):
        print(f"[{i}] Source: {src}")
        print(doc)
        print()

    print("--- PLAN ---")
    print(result["plan"])
    print()

    print("--- DRAFT_ANSWER ---")
    print(result["draft_answer"])
    print()

    print("--- CRITIC_FEEDBACK ---")
    print(result["critic_feedback"])
    print()

    print("--- FINAL_ANSWER ---")
    print(result["final_answer"])
    print()


def main() -> None:
    # You can change this query to test different documents
    query = "Is biotech xyz a reliable company?"
    result = run_pipeline(query, k=3)
    pretty_print_pipeline(result)

if __name__ == "__main__":
    main()

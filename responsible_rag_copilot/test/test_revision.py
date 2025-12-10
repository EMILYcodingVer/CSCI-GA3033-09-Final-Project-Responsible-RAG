from revision import revise_answer


def main():
    query = "What is responsible AI?"
    retrieved_docs = [
        "Responsible AI refers to building AI systems that are safe, fair, transparent, and accountable."
    ]
    draft_answer = "Responsible AI means robots follow human laws and Asimov's fictional rules."
    critic_feedback = (
        "- Verdict: REVISE\n"
        "- Explanation: The answer mentions Asimov's fictional rules, which are not in the document.\n"
        "- Suggested Fix: Responsible AI refers to building AI systems that are safe, fair, transparent, and accountable."
    )

    final_answer = revise_answer(query, retrieved_docs, draft_answer, critic_feedback)

    print("\nFinal revised answer:\n")
    print(final_answer)


if __name__ == "__main__":
    main()
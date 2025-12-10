from critic import evaluate_answer

def main():
    query = "What is responsible AI?"
    retrieved_docs = [
        "Responsible AI refers to building AI systems that are safe, fair, transparent, and accountable."
    ]

    # A deliberately imperfect answer to test the critic
    draft_answer = "Responsible AI means robots will follow human laws and avoid harming people, like Asimov's rules."

    critique = evaluate_answer(query, retrieved_docs, draft_answer)

    print("\nCritic Output:\n")
    print(critique)


if __name__ == "__main__":
    main()
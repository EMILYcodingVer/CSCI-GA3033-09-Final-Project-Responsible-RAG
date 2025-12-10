from planner import generate_plan


def main():
    query = "What is responsible AI?"
    retrieved_docs = [
        "Responsible AI refers to building AI systems that are safe, fair, transparent, and accountable.",
        "OpenAI is an AI research and deployment company."
    ]

    plan = generate_plan(query, retrieved_docs)
    print("\nGenerated Plan:\n")
    print(plan)


if __name__ == "__main__":
    main()
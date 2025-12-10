from rag import SimpleRAG


def main() -> None:
    """
    Simple test script for the SimpleRAG retriever.

    It loads all .txt files under the `data/` directory,
    builds embeddings, and prints the top-k retrieved chunks
    for a sample query.
    """
    # Use the entire `data/` directory as the corpus
    rag = SimpleRAG(corpus_path="data")

    # Example query
    query = "According to the EU AI Act, what is a high-risk AI system?"
    results = rag.retrieve(query, k=3)

    print("Query:", query)
    print("Top documents:")

    for i, r in enumerate(results, start=1):
        print(f"\n--- Result {i} (similarity={r['similarity']:.4f}) ---")
        print(f"[Source: {r['source']}]")
        print(r["text"])


if __name__ == "__main__":
    main()
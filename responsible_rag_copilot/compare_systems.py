from typing import Dict, Any, List

from openai import OpenAI

from config import OPENAI_API_KEY, CHAT_MODEL
from rag import SimpleRAG
from answer_pipeline import run_pipeline  # full system pipeline

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Create a RAG instance for the simple RAG baseline
# (This will load and embed all .txt files under data/)
RAG = SimpleRAG(corpus_path="data")


def answer_llm_only(query: str) -> str:
    """
    Baseline 1: LLM-only, no retrieval.

    The model only sees the user question and answers directly
    from its pretraining, without access to your documents.
    """
    system_prompt = (
        "You are a helpful assistant answering questions about AI, "
        "responsible AI, AI regulations, and AI governance. "
        "Answer the user's question as best as you can, but you do NOT "
        "have access to any external documents."
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )

    return response.choices[0].message.content


def answer_simple_rag(query: str, k: int = 3) -> Dict[str, Any]:
    """
    Baseline 2: Simple RAG.

    Steps:
      1) Retrieve top-k chunks with SimpleRAG.
      2) Directly prompt the chat model with the question and the retrieved docs.
      3) No planner, no critic, no revision.

    Returns:
        A dict with:
        - "answer": model answer
        - "retrieved_docs": list of doc chunks (strings)
        - "sources": list of source ids (filename#chunk_index)
    """
    retrieved = RAG.retrieve(query, k=k)
    docs = [r["text"] for r in retrieved]
    sources = [r["source"] for r in retrieved]

    docs_text = "\n\n".join(
        f"Document {i+1} (source={src}):\n{doc}"
        for i, (doc, src) in enumerate(zip(docs, sources), start=1)
    )

    system_prompt = (
        "You are an assistant that answers questions about responsible AI, "
        "AI principles, and AI regulations.\n"
        "You are given a user question and several retrieved document chunks. "
        "Use ONLY the information in these documents to answer the question. "
        "If the documents are not sufficient, say that you are not sure "
        "instead of guessing."
    )

    user_content = (
        f"User question:\n{query}\n\n"
        f"Retrieved documents:\n{docs_text}\n\n"
        "Now answer the user's question based on the documents above."
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "retrieved_docs": docs,
        "sources": sources,
    }


def pretty_print_section(title: str) -> None:
    """Print a nice section header."""
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print()


def compare_systems(query: str, k: int = 3) -> None:
    """
    Run and compare three systems on the same query:

        1) LLM-only (no retrieval)
        2) Simple RAG (retrieval + direct answer)
        3) Full pipeline (RAG + planner + critic + revision)

    Prints all answers and some retrieval information to the console.
    """
    # 1) LLM-only
    llm_only_answer = answer_llm_only(query)

    # 2) Simple RAG
    simple_rag_result = answer_simple_rag(query, k=k)

    # 3) Full pipeline
    full_result = run_pipeline(query, k=k)

    # ----- Print results -----

    pretty_print_section("QUERY")
    print(query)

    # LLM-only
    pretty_print_section("BASELINE 1: LLM-ONLY (NO RETRIEVAL)")
    print(llm_only_answer)

    # Simple RAG
    pretty_print_section("BASELINE 2: SIMPLE RAG (RETRIEVAL + DIRECT ANSWER)")
    print("--- Retrieved sources ---")
    for src in simple_rag_result["sources"]:
        print(f"- {src}")
    print()
    print("--- Answer ---")
    print(simple_rag_result["answer"])

    # Full pipeline
    pretty_print_section("SYSTEM 3: FULL PIPELINE (RAG + PLANNER + CRITIC + REVISION)")
    print("--- Retrieved sources ---")
    unique_sources = sorted(set(full_result["sources"]))
    for src in unique_sources:
        print(f"- {src}")
    print()
    print("--- Plan ---")
    print(full_result["plan"])
    print()
    print("--- Draft answer ---")
    print(full_result["draft_answer"])
    print()
    print("--- Critic feedback ---")
    print(full_result["critic_feedback"])
    print()
    print("--- FINAL ANSWER ---")
    print(full_result["final_answer"])


def main() -> None:
    # You can change this query to test different aspects of your corpus
    query = "Is ChinaMobile a reliable company?"

    compare_systems(query, k=3)


if __name__ == "__main__":
    main()
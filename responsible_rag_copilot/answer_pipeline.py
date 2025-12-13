from typing import List, Dict, Any

from openai import OpenAI

from config import OPENAI_API_KEY, CHAT_MODEL
from rag import SimpleRAG
from planner import generate_plan
from critic import evaluate_answer
from revision import revise_answer

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Create a single global RAG instance so we do not recompute embeddings
RAG = SimpleRAG(corpus_path="data")


DRAFT_SYSTEM_PROMPT = """
You are an assistant that writes an initial answer to questions about
responsible AI, AI principles, AI regulations, and AI governance.

You are given:
- the user's question
- a set of retrieved document chunks
- a high-level plan

Your job is to write a clear, well-structured draft answer that:
- is grounded in the retrieved documents,
- follows the plan,
- is safe and non-harmful,
- does not hallucinate facts.

If the retrieved documents do not contain enough information, say so explicitly.
"""


def generate_draft_answer(
    query: str,
    retrieved_docs: List[str],
    plan: str,
) -> str:
    """
    Generate a draft answer using the query, retrieved documents, and the plan.

    Args:
        query          : user question
        retrieved_docs : list of retrieved document chunks
        plan           : reasoning / action plan produced by the planner

    Returns:
        Draft answer as a string.
    """
    docs_text = "\n\n".join(
        f"Document {i+1}: {d}" for i, d in enumerate(retrieved_docs)
    )

    user_content = (
        f"User question:\n{query}\n\n"
        f"Retrieved documents:\n{docs_text}\n\n"
        f"Plan:\n{plan}\n\n"
        "Now follow the plan and write a draft answer grounded in the documents."
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": DRAFT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    return response.choices[0].message.content


def run_pipeline(query: str, k: int = 3) -> Dict[str, Any]:
    """
    Run the full responsible RAG pipeline for a single query.

    Steps:
        1) Retrieve top-k documents with SimpleRAG.
        2) Generate a high-level plan.
        3) Generate a draft answer following the plan.
        4) Ask the critic to evaluate safety and factual grounding.
        5) If needed, ask the revision agent to revise the draft answer.

    Args:
        query : user question string
        k     : number of documents to retrieve

    Returns:
        A dict with keys:
        - "query"
        - "retrieved_docs"
        - "sources"
        - "plan"
        - "draft_answer"
        - "critic_feedback"
        - "final_answer"
    """
    # 1) Retrieve
    retrieved = RAG.retrieve(query, k=k)
    retrieved_texts = [r["text"] for r in retrieved]
    retrieved_sources = [r["source"] for r in retrieved]

    # 2) Planner
    plan = generate_plan(query, retrieved_texts)

    # 3) Draft answer
    draft_answer = generate_draft_answer(query, retrieved_texts, plan)

    # 4) Critic
    critic_feedback = evaluate_answer(query, retrieved_texts, draft_answer)

    # 5) Decide whether to revise
    critic_upper = critic_feedback.upper()
    needs_revision = "VERDICT: REVISE" in critic_upper

    if needs_revision:
        final_answer = revise_answer(
            query=query,
            retrieved_docs=retrieved_texts,
            draft_answer=draft_answer,
            critic_feedback=critic_feedback,
        )
    else:
        final_answer = draft_answer

    return {
        "query": query,
        "retrieved_docs": retrieved_texts,
        "sources": retrieved_sources,
        "plan": plan,
        "draft_answer": draft_answer,
        "critic_feedback": critic_feedback,
        "final_answer": final_answer,
    }

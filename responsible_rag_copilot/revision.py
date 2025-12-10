from typing import List
from openai import OpenAI

from config import OPENAI_API_KEY, CHAT_MODEL

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

REVISION_SYSTEM_PROMPT = """
You are a careful assistant that revises answers about responsible AI and AI governance.

Your job is to:
1) Read the user's question.
2) Read the retrieved documents (which are the ground truth).
3) Read the draft answer.
4) Read the critic's feedback.

Then you must produce a FINAL REVISED ANSWER that:
- is fully grounded in the retrieved documents (do not invent facts),
- is safe and non-harmful,
- is concise but complete,
- directly addresses the user's question.

Do NOT mention the existence of planner, critic, or revision agents.
Just answer the question clearly.
"""


def revise_answer(
    query: str,
    retrieved_docs: List[str],
    draft_answer: str,
    critic_feedback: str,
) -> str:
    """
    Revise the draft answer using critic feedback and retrieved documents.

    Args:
        query           : user question
        retrieved_docs  : list of top-k retrieved document chunks (strings)
        draft_answer    : initial draft answer from the model
        critic_feedback : feedback string returned by the critic

    Returns:
        A revised final answer as a string.
    """
    docs_text = "\n\n".join(
        f"Document {i+1}: {d}" for i, d in enumerate(retrieved_docs)
    )

    user_content = (
        f"User question:\n{query}\n\n"
        f"Retrieved documents (ground truth):\n{docs_text}\n\n"
        f"Draft answer:\n{draft_answer}\n\n"
        f"Critic feedback:\n{critic_feedback}\n\n"
        "Please produce a revised final answer that follows the critic's feedback "
        "and stays grounded in the retrieved documents."
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": REVISION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    return response.choices[0].message.content
from openai import OpenAI
from typing import List

from config import OPENAI_API_KEY, CHAT_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


PLANNER_SYSTEM_PROMPT = """
You are a reasoning planner. Your job is to produce a clean, structured plan that explains how to answer the user's question using the retrieved documents.

Rules:
- Do NOT produce the final answer.
- Only produce a plan (Thought + Steps).
- Be concise, logical, and helpful.
- Do not hallucinate content not found in the retrieved documents.
"""


def generate_plan(query: str, retrieved_docs: List[str]) -> str:
    """
    Generate a reasoning plan given the query and retrieved documents.
    """
    docs_text = "\n\n".join(f"Document {i+1}: {d}" for i, d in enumerate(retrieved_docs))

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": f"User question:\n{query}\n\nRetrieved documents:\n{docs_text}"}
        ]
    )

    return response.choices[0].message.content
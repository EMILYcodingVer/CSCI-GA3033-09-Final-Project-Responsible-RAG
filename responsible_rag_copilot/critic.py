from openai import OpenAI
from typing import List

from config import OPENAI_API_KEY, CHAT_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

CRITIC_SYSTEM_PROMPT = """You are a safety and factuality critic for a retrieval-augmented generation (RAG) system.

Your job is to evaluate the model's draft answer and determine:

- how well it is grounded in the retrieved documents
- whether it is safe and non-harmful
- whether it contains hallucinations or speculative claims
- what issues the user should be aware of

You MUST respond with a single valid JSON object and nothing else.
Do NOT include markdown, comments, or any text outside the JSON.

Use the following JSON schema exactly:

{
  "overall_score": float,          // 0.0–1.0, overall reliability of the answer
  "grounding_score": float,        // 0.0–1.0, how well the answer is supported by the documents
  "safety_score": float,           // 0.0–1.0, how safe and non-harmful the answer is
  "hallucination_risk": "low" | "medium" | "high",
  "issues": [string],              // short bullet-style descriptions of problems or uncertainties
  "suggestions": [string],         // short suggestions to improve or safely interpret the answer
  "summary": string                // 1–3 sentences summarizing your assessment in plain language
}

Scoring & risk calibration (IMPORTANT):
- Treat 0.8–1.0 as HIGH score, 0.5–0.79 as MEDIUM, below 0.5 as LOW.
- "hallucination_risk" must be consistent with the scores:
    * Use "low" when grounding_score >= 0.8 and overall_score >= 0.7.
    * Use "medium" when scores are moderate or mixed.
    * Use "high" when grounding_score < 0.5 or you see major unsupported/speculative content.

Guidelines:
- "issues" can include lack of evidence, outdated data, speculative claims, missing perspectives, etc.
- If the answer is strong, "issues" can be an empty list but must still be present.
- If the answer is unsafe or speculative, lower the scores and explain why.
- Always fill in ALL fields in the JSON.
"""


def evaluate_answer(query: str, retrieved_docs: List[str], draft_answer: str) -> str:
    """
    Evaluate the draft answer given the user's query and retrieved documents.

    Returns a JSON string following the schema described in CRITIC_SYSTEM_PROMPT.
    """
    docs_text = "\n\n".join(f"Document {i+1}: {d}" for i, d in enumerate(retrieved_docs))

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,  # make critic as deterministic/stable as possible
        top_p=0,
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User question:\n{query}\n\n"
                    f"Retrieved documents:\n{docs_text}\n\n"
                    f"Draft answer:\n{draft_answer}"
                ),
            },
        ],
    )

    # The content is a JSON string, which app_new.py will parse with parse_critic_json()
    return response.choices[0].message.content

from openai import OpenAI
from config import OPENAI_API_KEY, CHAT_MODEL

# Create OpenAI client using the loaded API key
client = OpenAI(api_key=OPENAI_API_KEY)

def chat_once(system_prompt: str, user_message: str, temperature: float = 0.2) -> str:
    """
    Call the OpenAI chat completion API once and return the assistant's reply text.
    """
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",  "content": user_message},
        ],
    )
    return response.choices[0].message.content
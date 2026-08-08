"""
Narrator LLM call, async so a slow response from one room doesn't stall
the websocket event loop for every other room.

Uses Gemini via the Google Generative AI API.
"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


async def call_narrator(system_prompt: str, user_text: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "No GEMINI_API_KEY set on the server. Add it to backend/.env "
            "(see .env.example) and restart the server."
        )

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [
                    {
                        "parts": [
                            {"text": f"{system_prompt}\n\nUser input:\n{user_text}"}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.8,
                    "maxOutputTokens": 1000,
                },
            },
        )

    if response.status_code >= 400:
        if response.status_code in (401, 403):
            raise RuntimeError(f"Gemini API key was rejected ({response.status_code}).")
        if response.status_code == 429:
            return (
                "The narrative engine is in fallback mode because the Gemini quota has been exhausted. "
                "The room remains active, and the next decision will continue the story in a simple, "
                "direct manner."
            )
        raise RuntimeError(f"Gemini request failed: {response.status_code} {response.text[:300]}")

    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return ""


# --- To use Anthropic's Claude instead of DeepSeek ---------------------------
# import anthropic
# ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
# async def call_narrator(system_prompt: str, user_text: str) -> str:
#     msg = await client.messages.create(
#         model="claude-sonnet-4-6",
#         max_tokens=1000,
#         system=system_prompt,
#         messages=[{"role": "user", "content": user_text}],
#     )
#     return "".join(block.text for block in msg.content if block.type == "text")
# -------------------------------------------------------------------------------

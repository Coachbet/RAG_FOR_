"""LLM adapter: Groq API (free tier, fast). The key is read from the GROQ_API_KEY environment variable,
which lives in `.env` locally and in the Space "Secrets" on Hugging Face: never in the code or the repo.

Temperature 0 so that the evaluation is reproducible (Session 4.2).
"""
from __future__ import annotations

import time


class GroqLLM:
    def __init__(self, model: str = "llama-3.3-70b-versatile", api_key: str | None = None,
                 temperature: float = 0.0, max_retries: int = 5):
        from groq import Groq

        self.name = f"groq/{model}"
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = Groq(api_key=api_key) if api_key else Groq()

    def generate(self, system: str, user: str, max_tokens: int = 500) -> str:
        import groq

        for attempt in range(self.max_retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
                return (r.choices[0].message.content or "").strip()
            except groq.RateLimitError:  # free tier: a few requests per minute, wait and retry
                time.sleep(min(60, 5 * 2 ** attempt))
        raise RuntimeError("Groq rate limit: too many retries")

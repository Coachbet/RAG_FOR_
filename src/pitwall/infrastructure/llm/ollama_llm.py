"""LLM adapter: a local model served by Ollama (https://ollama.com), e.g. `ollama pull qwen2.5:3b`.

Ollama is a runtime, Qwen is a model: Ollama can run Qwen, Llama, Mistral... Useful to compare a small
local model with the Groq API in the report, and to keep data on the machine. Not used on the HF Space
(free CPU Space: too slow).
"""
from __future__ import annotations

import httpx


class OllamaLLM:
    def __init__(self, model: str = "qwen2.5:3b", host: str = "http://localhost:11434", temperature: float = 0.0):
        self.name = f"ollama/{model}"
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature

    def generate(self, system: str, user: str, max_tokens: int = 500) -> str:
        r = httpx.post(
            f"{self.host}/api/chat",
            json={"model": self.model, "stream": False,
                  "options": {"temperature": self.temperature, "num_predict": max_tokens},
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

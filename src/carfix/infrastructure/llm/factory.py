"""Pick the LLM adapter from the configuration: LLM_PROVIDER = groq | ollama | transformers."""
from __future__ import annotations

from carfix.domain.ports import LLM


def make_llm(provider: str, model: str, api_key: str | None = None, ollama_host: str = "http://localhost:11434") -> LLM:
    provider = provider.lower()
    if provider == "groq":
        from .groq_llm import GroqLLM
        return GroqLLM(model=model, api_key=api_key)
    if provider == "ollama":
        from .ollama_llm import OllamaLLM
        return OllamaLLM(model=model, host=ollama_host)
    if provider in ("transformers", "hf", "local"):
        from .transformers_llm import TransformersLLM
        return TransformersLLM(model=model)
    raise ValueError(f"Unknown LLM_PROVIDER {provider!r} (expected groq, ollama or transformers)")

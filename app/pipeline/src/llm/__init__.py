"""LLMアダプタ層. プロバイダ抽象化と工場関数を提供."""

from __future__ import annotations

from .base import LLMClient, LLMResponse
from .prompts import build_extraction_prompt


def get_client(provider: str, model: str | None = None) -> LLMClient:
    """プロバイダ名から実装をディスパッチ.

    provider: "claude" | "gemini" | "openai"
    """
    provider = provider.lower()
    if provider == "claude":
        from .claude import ClaudeClient
        return ClaudeClient(model=model)
    if provider == "gemini":
        from .gemini import GeminiClient
        return GeminiClient(model=model)
    if provider in {"openai", "gpt", "gpt4o"}:
        from .openai_ import OpenAIClient
        return OpenAIClient(model=model)
    raise ValueError(f"Unknown provider: {provider}")


__all__ = ["LLMClient", "LLMResponse", "get_client", "build_extraction_prompt"]

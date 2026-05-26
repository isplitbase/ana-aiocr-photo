"""OpenAI GPT-4o / GPT-4.1 マルチモーダルアダプタ."""

from __future__ import annotations

import base64
import os
from typing import Optional

from PIL import Image

from .base import LLMClient, LLMResponse

DEFAULT_MODEL = "gpt-4o"

_PRICES_PER_MTOK = {
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
}


class OpenAIClient(LLMClient):
    name = "openai"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        try:
            from openai import OpenAI  # noqa: WPS433
        except ImportError as e:
            raise RuntimeError("openai SDK が未インストール. `pip install openai`") from e

        self._OpenAI = OpenAI
        self._model = model or DEFAULT_MODEL
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def extract(
        self,
        image: Image.Image,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        png = self._image_to_png_bytes(image)
        b64 = base64.standard_b64encode(png).decode("ascii")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
            ],
        })

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        in_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0
        cost = self._estimate_cost(in_tok, out_tok)
        return LLMResponse(
            text=text,
            model=self._model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
        )

    def _estimate_cost(self, in_tok: int, out_tok: int) -> float:
        prices = _PRICES_PER_MTOK.get(self._model, (2.5, 10.0))
        return (in_tok / 1_000_000) * prices[0] + (out_tok / 1_000_000) * prices[1]

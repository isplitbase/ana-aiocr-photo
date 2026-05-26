"""Anthropic Claude マルチモーダルアダプタ (デフォルト主軸)."""

from __future__ import annotations

import base64
import os
from typing import Optional

from PIL import Image

from .base import LLMClient, LLMResponse

DEFAULT_MODEL = "claude-sonnet-4-5"

# 概算単価 (USD / 1M tokens). 厳密な請求額計算ではなく目安.
_PRICES_PER_MTOK = {
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-opus-4-6": (15.0, 75.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


class ClaudeClient(LLMClient):
    name = "claude"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        try:
            import anthropic  # noqa: WPS433
        except ImportError as e:
            raise RuntimeError(
                "anthropic SDK が未インストール. `pip install anthropic`"
            ) from e

        self._anthropic = anthropic
        self._model = model or DEFAULT_MODEL
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )

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

        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        if system:
            kwargs["system"] = system

        msg = self._client.messages.create(**kwargs)
        # text content blocksを連結
        text = "".join(
            getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text"
        )
        usage = getattr(msg, "usage", None)
        in_tok = getattr(usage, "input_tokens", 0) if usage else 0
        out_tok = getattr(usage, "output_tokens", 0) if usage else 0
        cost = self._estimate_cost(in_tok, out_tok)
        return LLMResponse(
            text=text,
            model=self._model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
        )

    def _estimate_cost(self, in_tok: int, out_tok: int) -> float:
        prices = _PRICES_PER_MTOK.get(self._model, (3.0, 15.0))
        return (in_tok / 1_000_000) * prices[0] + (out_tok / 1_000_000) * prices[1]

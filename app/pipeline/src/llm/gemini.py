"""Google Gemini マルチモーダルアダプタ."""

from __future__ import annotations

import os
from typing import Optional

from PIL import Image

from .base import LLMClient, LLMResponse

DEFAULT_MODEL = "gemini-2.0-flash"

_PRICES_PER_MTOK = {
    "gemini-2.0-flash": (0.1, 0.4),
    "gemini-2.0-pro": (1.25, 5.0),
    "gemini-1.5-pro": (1.25, 5.0),
    "gemini-1.5-flash": (0.075, 0.3),
}


class GeminiClient(LLMClient):
    name = "gemini"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        try:
            from google import genai  # noqa: WPS433
        except ImportError as e:
            raise RuntimeError(
                "google-genai SDK が未インストール. `pip install google-genai`"
            ) from e

        self._genai = genai
        self._model = model or DEFAULT_MODEL
        self._client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))

    def extract(
        self,
        image: Image.Image,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        from google.genai import types  # noqa: WPS433

        png = self._image_to_png_bytes(image)
        contents = [
            types.Part.from_bytes(data=png, mime_type="image/png"),
            prompt,
        ]
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system,
            response_mime_type="application/json",
        )
        resp = self._client.models.generate_content(
            model=self._model, contents=contents, config=config,
        )
        text = getattr(resp, "text", "") or ""
        usage = getattr(resp, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
        out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0
        cost = self._estimate_cost(in_tok, out_tok)
        return LLMResponse(
            text=text,
            model=self._model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
        )

    def _estimate_cost(self, in_tok: int, out_tok: int) -> float:
        prices = _PRICES_PER_MTOK.get(self._model, (0.1, 0.4))
        return (in_tok / 1_000_000) * prices[0] + (out_tok / 1_000_000) * prices[1]

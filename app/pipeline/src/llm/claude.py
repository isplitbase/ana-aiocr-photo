"""Anthropic Claude マルチモーダルアダプタ (デフォルト主軸).

プロンプトキャッシュ対応 (2026-05-26):
  - システムプロンプト (全ページ共通の静的部分) に cache_control を付与し、
    連続処理時の入力コストを削減 (キャッシュ読込は通常入力の約10%課金)。
  - 画像は毎ページ異なるためキャッシュ対象外。user メッセージは画像先頭のまま
    (OCR品質維持) とし、system のみキャッシュする安全構成。
  - cache 書込み(1.25x)/読込(0.10x) を含めてコストを概算。
"""

from __future__ import annotations

import base64
import os
from typing import Optional

from PIL import Image

from .base import LLMClient, LLMResponse
from .model_resolver import resolve_model

DEFAULT_MODEL = "claude-sonnet-4-5"

# 概算単価 (USD / 1M tokens). 厳密な請求額計算ではなく目安.
_PRICES_PER_MTOK = {
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (15.0, 75.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

# temperature パラメータを受け付けないモデル (Opus 4.7 で廃止)
_NO_TEMPERATURE = ("opus-4-7",)

# プロンプトキャッシュの倍率 (通常入力単価に対する比)
_CACHE_WRITE_MULT = 1.25  # 初回キャッシュ書込み
_CACHE_READ_MULT = 0.10   # キャッシュヒット読込


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
        self._model = resolve_model("claude", self._model, client=self._client)

    def extract(
        self,
        image: Image.Image,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        import io as _io
        # ★ APIの5MB上限は「base64エンコード後」のサイズに適用される。
        #    base64は約4/3倍に膨らむため、生バイトではなく b64長で判定する。
        _MAX_B64 = 5_242_880  # 5 MiB (base64文字列の上限)

        def _b64len(_b: bytes) -> int:
            return (len(_b) + 2) // 3 * 4

        _rgb = image.convert("RGB")
        _buf = _io.BytesIO()
        _rgb.save(_buf, format="PNG", optimize=True)
        img_bytes = _buf.getvalue()
        if _b64len(img_bytes) > _MAX_B64:
            # PNGがb64で5MB超 → JPEG高品質から段階的に下げる
            for _q in (92, 88, 82, 75, 68):
                _buf = _io.BytesIO()
                _rgb.save(_buf, format="JPEG", quality=_q)
                img_bytes = _buf.getvalue()
                if _b64len(img_bytes) <= _MAX_B64:
                    break
            else:
                # まだ超過 → 解像度を段階縮小 (最後の手段)
                _im2 = _rgb
                for _ in range(6):
                    _w, _h = _im2.size
                    _im2 = _im2.resize((int(_w * 0.9), int(_h * 0.9)), Image.LANCZOS)
                    _buf = _io.BytesIO()
                    _im2.save(_buf, format="JPEG", quality=80)
                    img_bytes = _buf.getvalue()
                    if _b64len(img_bytes) <= _MAX_B64:
                        break
        b64 = base64.standard_b64encode(img_bytes).decode("ascii")
        media_type = "image/png" if img_bytes[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"

        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        # 静的な抽出指示を先頭に置き cache_control を付与
                        # (画像はページ毎に異なるためキャッシュ対象外。画像は末尾)
                        {
                            "type": "text",
                            "text": prompt,
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        },
                    ],
                }
            ],
        }
        # Opus 4.7 等は temperature 廃止。対応モデルのみ付与。
        if not any(tag in self._model for tag in _NO_TEMPERATURE):
            kwargs["temperature"] = temperature
        # システムプロンプトはプロンプトキャッシュ対象 (全ページ共通の静的部分)
        if system:
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        msg = self._client.messages.create(**kwargs)
        # text content blocksを連結
        text = "".join(
            getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text"
        )
        usage = getattr(msg, "usage", None)
        in_tok = getattr(usage, "input_tokens", 0) if usage else 0
        out_tok = getattr(usage, "output_tokens", 0) if usage else 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0 if usage else 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0 if usage else 0
        cost = self._estimate_cost(in_tok, out_tok, cache_write, cache_read)
        return LLMResponse(
            text=text,
            model=self._model,
            input_tokens=in_tok + cache_write + cache_read,
            output_tokens=out_tok,
            cost_usd=cost,
        )

    def _estimate_cost(self, in_tok: int, out_tok: int,
                       cache_write: int = 0, cache_read: int = 0) -> float:
        prices = _PRICES_PER_MTOK.get(self._model)
        if prices is None:
            for k, v in _PRICES_PER_MTOK.items():
                if k in self._model or self._model in k:
                    prices = v
                    break
        if prices is None:
            prices = (5.0, 25.0) if "opus" in self._model else (3.0, 15.0)
        in_price, out_price = prices
        return (
            in_tok / 1_000_000 * in_price
            + cache_write / 1_000_000 * in_price * _CACHE_WRITE_MULT
            + cache_read / 1_000_000 * in_price * _CACHE_READ_MULT
            + out_tok / 1_000_000 * out_price
        )

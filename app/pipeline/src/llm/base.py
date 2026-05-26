"""LLMアダプタ基底クラス."""

from __future__ import annotations

import abc
import io
from dataclasses import dataclass
from typing import Optional

from PIL import Image


@dataclass
class LLMResponse:
    """LLM呼び出し結果."""

    text: str  # LLMの生レスポンス本文 (JSON文字列を想定)
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Optional[float] = None


class LLMClient(abc.ABC):
    """マルチモーダルLLMの最小インターフェース.

    画像 + プロンプトを送り, テキストを得るだけ.
    JSON抽出やリトライはパイプライン側で行う.
    """

    name: str = "abstract"

    @abc.abstractmethod
    def extract(
        self,
        image: Image.Image,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        ...

    @staticmethod
    def _image_to_png_bytes(image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG", optimize=True)
        return buf.getvalue()

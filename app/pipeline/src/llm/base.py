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
    def _image_to_png_bytes(image: Image.Image, max_b64: int = 5_242_880) -> bytes:
        """画像をバイト列に. API上限 (base64で5MB) を超える場合はJPEG化/段階縮小で収める.

        ★ API上限は base64エンコード後のサイズに適用される (生バイト×約4/3)。
          そのため b64長で判定する。
        高解像度 (鮮明さ) を保つため、まずPNG -> 超過ならJPEG高品質 ->
        それでも超過なら品質/解像度を段階的に下げて上限以内にする。
        """
        def _b64len(b: bytes) -> int:
            return (len(b) + 2) // 3 * 4

        rgb = image.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
        if _b64len(data) <= max_b64:
            return data

        # PNGが大きすぎる -> JPEGで高品質エンコード (鮮明さ維持しつつ大幅減量)
        for quality in (92, 88, 82, 75, 68):
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=quality)
            data = buf.getvalue()
            if _b64len(data) <= max_b64:
                return data

        # まだ超過 -> 解像度を段階的に下げる (最後の手段)
        img2 = rgb
        for _ in range(6):
            w, h = img2.size
            img2 = img2.resize((int(w * 0.85), int(h * 0.85)), Image.LANCZOS)
            buf = io.BytesIO()
            img2.save(buf, format="JPEG", quality=85)
            data = buf.getvalue()
            if _b64len(data) <= max_b64:
                return data
        return data  # 何回縮小しても超過ならそのまま返す (稀)

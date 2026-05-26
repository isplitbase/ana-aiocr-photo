"""適応的解像度ローダー (独立モジュール).

大判PDF (例: 5100x6801px) を固定DPIで巨大レンダしてから縮小する無駄を回避し、
各ページの実寸から「長辺が target_long_side px になるDPI」を逆算してレンダする。
これにより大判ファイルでも最初から適切なサイズで高速・省メモリに読み込める。
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image, ImageOps

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def cap_resolution(img: Image.Image, max_side: int = 2200) -> Image.Image:
    """長辺を抑える (拡大はしない)."""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / longest
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def load_as_pages_adaptive(
    path: str | Path,
    target_long_side: int = 1600,
    max_dpi: int = 300,
    min_dpi: int = 72,
) -> List[Image.Image]:
    """適応的解像度でページ画像を読み込む.

    Args:
        path: PDF or 画像ファイル
        target_long_side: 目標長辺ピクセル (読取精度の下限)
        max_dpi: レンダリングDPIの上限 (極端な拡大防止)
        min_dpi: レンダリングDPIの下限

    Returns:
        各ページ RGB の PIL Image リスト
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    ext = p.suffix.lower()
    if ext in SUPPORTED_IMAGE_EXTS:
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        return [cap_resolution(img, max_side=target_long_side)]

    if ext != ".pdf":
        raise ValueError(f"Unsupported file type: {ext}")

    import pypdfium2 as pdfium  # noqa: WPS433

    pdf = pdfium.PdfDocument(str(p))
    images = []
    for page in pdf:
        w_pt, h_pt = page.get_size()  # points (1/72 inch)
        long_pt = max(w_pt, h_pt, 1.0)
        dpi = target_long_side * 72.0 / long_pt
        dpi = max(min_dpi, min(dpi, max_dpi))
        scale = dpi / 72.0
        pil = page.render(scale=scale).to_pil().convert("RGB")
        pil = cap_resolution(pil, max_side=target_long_side)
        images.append(pil)
    return images

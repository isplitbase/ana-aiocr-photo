"""PDF入出力ユーティリティ.

- PDF / 画像ファイルを「ページ画像のリスト」として正規化する
- pdf2image (poppler) を優先, 失敗時は pypdfium2 にフォールバック
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image, ImageOps


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_as_pages(path: str | Path, dpi: int = 220) -> List[Image.Image]:
    """ファイルを Pillow Image のリストとして読み込む.

    Args:
        path: PDF or 画像ファイル
        dpi: PDF→画像変換時のDPI (LLM入力向けに220前後を推奨)

    Returns:
        各ページ/各画像を表す RGB の PIL Image のリスト
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    ext = p.suffix.lower()
    if ext in SUPPORTED_IMAGE_EXTS:
        img = Image.open(p)
        img = ImageOps.exif_transpose(img)  # スマホ撮影の自動回転
        return [img.convert("RGB")]

    if ext == ".pdf":
        return _pdf_to_images(p, dpi=dpi)

    raise ValueError(f"Unsupported file type: {ext}")


def _pdf_to_images(path: Path, dpi: int) -> List[Image.Image]:
    """PDF→Imageリスト. pdf2image を優先, 失敗時 pypdfium2."""
    try:
        from pdf2image import convert_from_path

        return [img.convert("RGB") for img in convert_from_path(str(path), dpi=dpi)]
    except Exception:
        import pypdfium2 as pdfium  # noqa: WPS433

        pdf = pdfium.PdfDocument(str(path))
        scale = dpi / 72.0
        images = []
        for page in pdf:
            pil = page.render(scale=scale).to_pil().convert("RGB")
            images.append(pil)
        return images


def cap_resolution(img: Image.Image, max_side: int = 2200) -> Image.Image:
    """LLM入力前に長辺を抑える (コスト/レイテンシ削減)."""
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

    大判PDF (例: 5100x6801px) を固定DPIで巨大レンダしてから縮小する無駄を回避し、
    各ページの実寸から「長辺が target_long_side px になるDPI」を逆算してレンダする。
    これにより大判ファイルでも最初から適切なサイズで高速・省メモリに読み込める。

    - 元が target_long_side より小さいページは拡大しない (min_dpi下限あり)
    - 「読取れるギリギリ」= target_long_side を読取精度の下限とする

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
        # 画像ファイルは長辺が target を超える場合のみ縮小 (拡大しない)
        return [cap_resolution(img, max_side=target_long_side)]

    if ext != ".pdf":
        raise ValueError(f"Unsupported file type: {ext}")

    import pypdfium2 as pdfium  # noqa: WPS433

    pdf = pdfium.PdfDocument(str(p))
    images = []
    for page in pdf:
        w_pt, h_pt = page.get_size()  # points (1/72 inch)
        long_pt = max(w_pt, h_pt, 1.0)
        # 目標長辺pxになるDPI: target = long_pt/72 * dpi  →  dpi = target*72/long_pt
        dpi = target_long_side * 72.0 / long_pt
        dpi = max(min_dpi, min(dpi, max_dpi))
        scale = dpi / 72.0
        pil = page.render(scale=scale).to_pil().convert("RGB")
        # 念のため長辺を target に揃える (DPI上限で超過した場合のみ縮小)
        pil = cap_resolution(pil, max_side=target_long_side)
        images.append(pil)
    return images

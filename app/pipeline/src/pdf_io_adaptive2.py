"""適応的解像度ローダー v2 (高DPIオーバーサンプル→LANCZOS鮮明縮小方式).

【識字率低下の修正 (2026-05-25)】
旧方式の問題:
  - 旧adaptive: 目標pxになるDPIを逆算して「そのDPIで直接レンダ」→ A4で約139DPI相当の
    低DPIレンダになり文字が滲んでいた (識字率低下の主因)。
  - adaptive2初版: 長辺2200pxへ引き上げたが、Haiku/Sonnetは長辺1568pxへ自動縮小するため
    精度に寄与せず、かつ2200px PNGが5MB超でAPIエラーになっていた。

新方式 (本ファイル):
  1. 高DPI (render_dpi=300相当) でオーバーサンプリング描画 → 文字輪郭が鮮明
  2. LANCZOSで long edge = target_long_side に高品質縮小
  3. target_long_side はモデルのネイティブ解像度に合わせる:
       - Haiku / Sonnet 系: 1568 (API側の自動縮小と一致。これ以上送っても無意味)
       - Opus 4.x 系      : 2576
  4. 大判PDFは max_render_long_side で描画上限を設け、巨大レンダ/メモリ膨張を回避
送信時のJPEG化 (5MB以内) は src/llm/base.py・claude.py 側で担保。
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image, ImageOps

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# モデル系列ごとのネイティブ長辺 (API側で自動縮小される上限)
MODEL_NATIVE_LONG_SIDE = {
    "haiku": 1568,
    "sonnet": 1568,
    "opus": 2576,
}


def native_long_side_for_model(model):
    """モデル名から最適な送信長辺pxを返す (不明時は1568)."""
    if not model:
        return 1568
    m = model.lower()
    for key, px in MODEL_NATIVE_LONG_SIDE.items():
        if key in m:
            return px
    return 1568


def cap_resolution(img: Image.Image, max_side: int) -> Image.Image:
    """長辺が max_side を超える場合のみ LANCZOS で縮小 (拡大はしない)."""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / longest
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def load_as_pages_adaptive(
    path,
    target_long_side: int = 1568,
    render_dpi: int = 300,
    max_render_long_side: int = 4000,
) -> List[Image.Image]:
    """高DPIで描画してから target_long_side へ鮮明縮小したページ画像リストを返す.

    Args:
        path: PDF or 画像ファイル
        target_long_side: 最終送信時の長辺px (モデルのネイティブ解像度に合わせる)
        render_dpi: PDF描画DPI (オーバーサンプル用。300推奨)
        max_render_long_side: 描画長辺の上限px (大判PDFの巨大レンダ防止)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    ext = p.suffix.lower()
    if ext in SUPPORTED_IMAGE_EXTS:
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        return [cap_resolution(img, max_side=target_long_side)]

    if ext != ".pdf":
        raise ValueError("Unsupported file type: " + ext)

    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(p))
    images = []
    for page in pdf:
        w_pt, h_pt = page.get_size()
        long_pt = max(w_pt, h_pt, 1.0)

        # 描画長辺 = render_dpi 相当。上限と「target未満にしない」を保証。
        render_long = long_pt / 72.0 * render_dpi
        render_long = min(render_long, float(max_render_long_side))
        render_long = max(render_long, float(target_long_side) * 1.05)

        dpi = render_long * 72.0 / long_pt
        scale = dpi / 72.0
        pil = page.render(scale=scale).to_pil().convert("RGB")
        pil = cap_resolution(pil, max_side=target_long_side)
        images.append(pil)
    return images


def _main():
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m src.pdf_io_adaptive2 PDF [target_long_side]", file=sys.stderr)
        sys.exit(2)
    tgt = int(sys.argv[2]) if len(sys.argv) > 2 else 1568
    imgs = load_as_pages_adaptive(sys.argv[1], target_long_side=tgt)
    for i, im in enumerate(imgs, 1):
        print("page " + str(i) + ": " + str(im.size[0]) + "x" + str(im.size[1]))


if __name__ == "__main__":
    _main()

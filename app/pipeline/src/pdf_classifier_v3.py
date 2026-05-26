"""PDF種別判定 v3: text_pdf / scan_pdf / photo_pdf.

v2 + 適応的解像度:
  ★ 判定用画像を load_as_pages_adaptive で読み込み、大判PDFでも長辺を
    一定に抑えて高速判定する (大判の巨大レンダを回避)。

判定ロジック (v2継承):
    1. photo_score を必ず先に計算 (テキスト層の有無に関わらず)
    2. photo_score >= 閾値 → photo_pdf
    3. photo_score < 閾値 かつ テキスト豊富&高品質 → text_pdf
    4. それ以外 → scan_pdf

photo_score (0-1): 傾き / 照明ムラ / 背景多色性 / ページ縁 の4特徴 (各0.25)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class PDFKind(str, Enum):
    TEXT_PDF = "text_pdf"
    SCAN_PDF = "scan_pdf"
    PHOTO_PDF = "photo_pdf"
    UNKNOWN = "unknown"


@dataclass
class ClassificationReport:
    kind: PDFKind
    num_pages: int
    text_chars_per_page: float
    image_area_ratio: float
    photo_score: float
    photo_features: dict
    text_quality: dict
    notes: list[str]

    def to_dict(self):
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


# ---------------------------------------------------------------------------
# テキスト層品質
# ---------------------------------------------------------------------------

_ACCT_TERMS = [
    "資産", "負債", "純資産", "現金", "預金", "売掛金", "買掛金", "売上",
    "仕入", "合計", "利益", "費用", "資本", "流動", "固定", "棚卸",
    "手形", "未払", "未収", "繰越", "経費", "営業", "販売",
]


def text_layer_quality(text: str) -> dict:
    """テキスト層の品質を測る (num_quality = カンマ区切り数値のクリーン率)."""
    comma_tokens = re.findall(r'\S*[,，]\S*', text)
    clean_num = 0
    for t in comma_tokens:
        core = re.sub(r'[△▲\-()（）\[\]【】 ]', '', t)
        if core and re.fullmatch(r'[\d,，.]+', core):
            clean_num += 1
    num_quality = clean_num / max(len(comma_tokens), 1)
    term_hits = sum(1 for t in _ACCT_TERMS if t in text)
    return {
        "comma_tokens": len(comma_tokens),
        "num_quality": round(num_quality, 3),
        "term_hits": term_hits,
    }


def _analyze_pdf_structure(path: Path) -> tuple[float, float, int, dict]:
    """(avg_text_chars, image_area_ratio, num_pages, text_quality) を返す."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    n = len(pdf)
    if n == 0:
        return 0.0, 0.0, 0, {"comma_tokens": 0, "num_quality": 0.0, "term_hits": 0}

    total_chars = 0
    total_image_area_ratio = 0.0
    all_text = []
    for page in pdf:
        textpage = page.get_textpage()
        try:
            text = textpage.get_text_range() or ""
        finally:
            textpage.close()
        total_chars += len(text.strip())
        all_text.append(text)

        page_w, page_h = page.get_size()
        page_area = max(page_w * page_h, 1.0)
        img_area = 0.0
        for obj in page.get_objects(filter=(pdfium.raw.FPDF_PAGEOBJ_IMAGE,)):
            try:
                l, b, r, t = obj.get_pos()
                img_area += max(r - l, 0) * max(t - b, 0)
            except Exception:
                continue
        total_image_area_ratio += min(img_area / page_area, 1.0)

    tq = text_layer_quality("\n".join(all_text))
    return total_chars / n, total_image_area_ratio / n, n, tq


# ---------------------------------------------------------------------------
# 写真的特徴のスコアリング
# ---------------------------------------------------------------------------

def _detect_skew_angle(gray: np.ndarray, max_angle: float = 15.0) -> float:
    edges = cv2.Canny(gray, 60, 180, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 720, threshold=200,
        minLineLength=gray.shape[1] // 4, maxLineGap=20,
    )
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 == x1:
            continue
        deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -max_angle <= deg <= max_angle:
            angles.append(deg)
    if not angles:
        return 0.0
    return abs(float(np.median(angles)))


def _lighting_unevenness(L: np.ndarray) -> float:
    h, w = L.shape
    n = 6
    bh, bw = h // n, w // n
    means = []
    for i in range(n):
        for j in range(n):
            block = L[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            if block.size:
                means.append(block.mean())
    if not means:
        return 0.0
    return float(np.std(means))


def _background_colorfulness(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask_paper = (hsv[:, :, 2] > 180) & (hsv[:, :, 1] < 35)
    non_paper_ratio = 1.0 - mask_paper.mean()
    return float(non_paper_ratio)


def _page_edge_detected(bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    img_area = bgr.shape[0] * bgr.shape[1]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.25 or area > img_area * 0.95:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return True
    return False


def _photo_score(img: Image.Image) -> tuple[float, dict]:
    bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    skew = _detect_skew_angle(gray)
    light_std = _lighting_unevenness(L)
    bg_color = _background_colorfulness(bgr)
    edge_ok = _page_edge_detected(bgr)

    s_skew = 0.25 if skew > 0.5 else (skew / 0.5) * 0.25
    s_light = min(light_std / 25.0, 1.0) * 0.25
    s_bg = min(bg_color / 0.3, 1.0) * 0.25
    s_edge = 0.25 if edge_ok else 0.0

    total = s_skew + s_light + s_bg + s_edge
    features = {
        "skew_deg": round(skew, 2),
        "lighting_std": round(light_std, 2),
        "non_paper_ratio": round(bg_color, 3),
        "page_edge_detected": edge_ok,
        "score_skew": round(s_skew, 3),
        "score_lighting": round(s_light, 3),
        "score_background": round(s_bg, 3),
        "score_edge": round(s_edge, 3),
    }
    return total, features


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_pdf(
    path: str | Path,
    *,
    sample_pages: int = 3,
    text_threshold_chars: int = 300,
    image_ratio_threshold: float = 0.3,
    photo_threshold: float = 0.37,
    text_quality_threshold: float = 0.6,
    feature_long_side: int = 1100,
) -> ClassificationReport:
    """PDFを text_pdf / scan_pdf / photo_pdf に分類 (適応的解像度・品質チェック付き)."""
    p = Path(path)
    notes: list[str] = []

    avg_chars, img_ratio, n_pages, tq = _analyze_pdf_structure(p)
    num_quality = tq.get("num_quality", 0.0)
    notes.append(f"avg_chars={avg_chars:.0f}, image_area_ratio={img_ratio:.2f}, pages={n_pages}")
    notes.append(f"text_quality: num_quality={num_quality}, term_hits={tq.get('term_hits')}, "
                 f"comma_tokens={tq.get('comma_tokens')}")

    # ★ photo_score を必ず先に計算 (テキスト層の有無に関わらず) ★
    # 適応的解像度で読み込み (大判ファイルでも長辺を一定に抑えて高速判定)
    from .pdf_io_adaptive import load_as_pages_adaptive

    imgs = load_as_pages_adaptive(p, target_long_side=feature_long_side)
    target_imgs = imgs[: min(sample_pages, len(imgs))]
    scores = []
    feats_agg = {"per_page": []}
    for i, im in enumerate(target_imgs, 1):
        s, f = _photo_score(im)
        scores.append(s)
        feats_agg["per_page"].append({"page": i, "score": round(s, 3), **f})
    photo_score = float(np.mean(scores)) if scores else 0.0
    notes.append(f"photo_score (avg of first {len(target_imgs)} pages) = {photo_score:.2f}")

    text_is_clean = (
        num_quality >= text_quality_threshold
        or (tq.get("comma_tokens", 0) < 10 and tq.get("term_hits", 0) >= 8)
    )

    if photo_score >= photo_threshold:
        kind = PDFKind.PHOTO_PDF
        notes.append(f"photo_score>={photo_threshold} → photo_pdf (テキスト層の有無に関わらず)")
    elif avg_chars >= text_threshold_chars and img_ratio < image_ratio_threshold and text_is_clean:
        kind = PDFKind.TEXT_PDF
        notes.append("low photo_score & text-rich & clean → text_pdf")
    else:
        kind = PDFKind.SCAN_PDF
        if avg_chars >= text_threshold_chars and not text_is_clean:
            notes.append(f"テキスト層は多いが低品質(num_quality={num_quality}) かつ photo_score<{photo_threshold} → scan_pdf")
        else:
            notes.append(f"photo_score<{photo_threshold} → scan_pdf")

    return ClassificationReport(
        kind=kind,
        num_pages=n_pages,
        text_chars_per_page=avg_chars,
        image_area_ratio=img_ratio,
        photo_score=round(photo_score, 3),
        photo_features=feats_agg,
        text_quality=tq,
        notes=notes,
    )


def _main():
    import json
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m src.pdf_classifier_v3 PDF", file=sys.stderr)
        sys.exit(2)
    rep = classify_pdf(sys.argv[1])
    print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()

"""画像前処理 v2 (LAB空間でカラー保持 + 薄文字ブースト).

v1 (preprocess.py) からの変更点:
    - flatten_lighting: GRAY→GRAY2BGR を廃止し, LAB空間のL channelのみ
      補正することでカラー情報を保持. LLMが「グレースケールで判読困難」
      と判定する原因を解消.
    - boost_faint: 全体が薄い帳票(L平均が高い)に対してガンマ補正で
      文字を引き上げる新処理を追加.
    - run_default: 上記2点を組み込んだ標準フロー.

旧版 (preprocess.py) は互換のため残置.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# 共通ヘルパは v1 から流用
from .preprocess import (
    _pil_to_cv,
    _cv_to_pil,
    auto_rotate,
    detect_page,
    warp_to_rect,
    enhance_contrast,
)


def flatten_lighting(img: Image.Image, kernel: int = 51) -> Image.Image:
    """LABのL channelのみGaussian除算+CLAHE. 色情報は保持."""
    cv = _pil_to_cv(img)
    lab = cv2.cvtColor(cv, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    if kernel % 2 == 0:
        kernel += 1
    background = cv2.GaussianBlur(L, (kernel, kernel), 0)
    background = np.where(background == 0, 1, background).astype(np.float32)
    L_norm = (L.astype(np.float32) / background) * 180.0
    L_norm = np.clip(L_norm, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_final = clahe.apply(L_norm)
    merged = cv2.merge([L_final, a, b])
    return _cv_to_pil(cv2.cvtColor(merged, cv2.COLOR_LAB2BGR))


def boost_faint(img: Image.Image, gain: float = 1.4, gamma: float = 0.85) -> Image.Image:
    """薄い文字を引き上げる. flatten_lighting後でも判読困難な画像向け."""
    cv = _pil_to_cv(img)
    lab = cv2.cvtColor(cv, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    L_f = L.astype(np.float32) / 255.0
    L_f = np.power(L_f, gamma)
    L_f = np.clip(L_f * gain - (gain - 1) * 0.5, 0, 1)
    L_out = (L_f * 255).astype(np.uint8)
    merged = cv2.merge([L_out, a, b])
    return _cv_to_pil(cv2.cvtColor(merged, cv2.COLOR_LAB2BGR))


@dataclass
class PreprocessSteps:
    rotated: bool = False
    page_detected: bool = False
    warped: bool = False
    flattened: bool = False
    contrast: bool = False
    boosted: bool = False


def _is_faint(img: Image.Image, threshold: float = 165.0) -> bool:
    cv = _pil_to_cv(img)
    lab = cv2.cvtColor(cv, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    return float(L.mean()) > threshold


def run_default(
    img: Image.Image,
    *,
    do_warp: bool = True,
    do_flatten: bool = True,
    do_contrast: bool = True,
    auto_boost_faint: bool = True,
) -> tuple[Image.Image, PreprocessSteps]:
    steps = PreprocessSteps()

    rotated = auto_rotate(img)
    if rotated is not img:
        steps.rotated = True
    img = rotated

    if do_warp:
        quad = detect_page(img)
        if quad is not None:
            steps.page_detected = True
            try:
                img = warp_to_rect(img, quad)
                steps.warped = True
            except cv2.error:
                pass

    if do_flatten:
        img = flatten_lighting(img)
        steps.flattened = True

    if auto_boost_faint and _is_faint(img):
        img = boost_faint(img)
        steps.boosted = True

    if do_contrast and not steps.flattened and not steps.boosted:
        img = enhance_contrast(img)
        steps.contrast = True

    return img, steps

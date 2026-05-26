"""画像前処理 v3 - 薄い帳票向けにさらに強化.

v2 からの変更点:
    - boost_faint をよりアグレッシブに (gain 1.8, gamma 0.6)
    - unsharp_mask を追加 (薄い文字の輪郭強調)
    - run_default が faint判定された画像に boost + sharp を両方適用
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .preprocess import (
    _pil_to_cv,
    _cv_to_pil,
    auto_rotate,
    detect_page,
    warp_to_rect,
    enhance_contrast,
)
from .preprocess_v2 import flatten_lighting  # カラー保持版


def boost_faint_strong(
    img: Image.Image, gain: float = 1.8, gamma: float = 0.6,
) -> Image.Image:
    """薄文字に対するアグレッシブな引き上げ.

    gamma 0.6 で 明度0.5 → 0.66 になり, 中間明度の薄い文字が濃くなる.
    その後 gain で全体コントラストを伸長.
    """
    cv = _pil_to_cv(img)
    lab = cv2.cvtColor(cv, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    L_f = L.astype(np.float32) / 255.0
    L_f = np.power(L_f, gamma)
    L_f = np.clip(L_f * gain - (gain - 1) * 0.5, 0, 1)
    L_out = (L_f * 255).astype(np.uint8)
    merged = cv2.merge([L_out, a, b])
    return _cv_to_pil(cv2.cvtColor(merged, cv2.COLOR_LAB2BGR))


def unsharp_mask(img: Image.Image, radius: int = 3, amount: float = 1.2) -> Image.Image:
    """アンシャープマスクで文字輪郭を強調."""
    cv = _pil_to_cv(img)
    blur = cv2.GaussianBlur(cv, (0, 0), sigmaX=radius)
    sharp = cv2.addWeighted(cv, 1 + amount, blur, -amount, 0)
    return _cv_to_pil(sharp)


@dataclass
class PreprocessSteps:
    rotated: bool = False
    page_detected: bool = False
    warped: bool = False
    flattened: bool = False
    boosted: bool = False
    sharpened: bool = False
    contrast: bool = False


def _is_faint(img: Image.Image, threshold: float = 160.0) -> bool:
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
        img = boost_faint_strong(img)
        steps.boosted = True
        img = unsharp_mask(img)
        steps.sharpened = True

    if do_contrast and not steps.flattened and not steps.boosted:
        img = enhance_contrast(img)
        steps.contrast = True

    return img, steps

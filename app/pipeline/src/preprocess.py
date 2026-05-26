"""画像前処理モジュール (LLM入力向けの軽い補正).

bash経由で完全書き直し版. v2/v3で使うヘルパも含む.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image


def _pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def auto_rotate(img: Image.Image, max_angle: float = 15.0) -> Image.Image:
    cv = _pil_to_cv(img)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 720, threshold=200,
        minLineLength=gray.shape[1] // 4, maxLineGap=20,
    )
    if lines is None:
        return img
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 == x1:
            continue
        deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -max_angle <= deg <= max_angle:
            angles.append(deg)
    if not angles:
        return img
    angle = float(np.median(angles))
    if abs(angle) < 0.3:
        return img
    h, w = cv.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        cv, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return _cv_to_pil(rotated)


@dataclass
class PageQuad:
    corners: np.ndarray


def _order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)],
    ], dtype=np.float32)


def detect_page(img: Image.Image) -> Optional[PageQuad]:
    cv = _pil_to_cv(img)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    img_area = cv.shape[0] * cv.shape[1]
    best = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.25:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best = approx
            best_area = area
    if best is None:
        return None
    return PageQuad(corners=_order_corners(best.astype(np.float32)))


def warp_to_rect(img: Image.Image, quad: PageQuad) -> Image.Image:
    tl, tr, br, bl = quad.corners
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array([
        [0, 0], [width - 1, 0],
        [width - 1, height - 1], [0, height - 1],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad.corners, dst)
    cv = _pil_to_cv(img)
    warped = cv2.warpPerspective(cv, M, (width, height), flags=cv2.INTER_CUBIC)
    return _cv_to_pil(warped)


def flatten_lighting(img: Image.Image, kernel: int = 51) -> Image.Image:
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


def enhance_contrast(img: Image.Image) -> Image.Image:
    cv = _pil_to_cv(img)
    lab = cv2.cvtColor(cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return _cv_to_pil(cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR))


def boost_faint(img: Image.Image, gain: float = 1.4, gamma: float = 0.85) -> Image.Image:
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

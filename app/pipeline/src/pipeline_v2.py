"""AI-OCRパイプライン v2.

v1 (pipeline.py) からの変更点:
    - preprocess_v3 (薄文字強化) を使用
    - _normalize_number の maketrans バグ修正
    - run() に max_rechecks_per_page=0 等パラメータ追加 (デバッグ用)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from .llm import build_extraction_prompt, get_client
from .llm.base import LLMResponse
from .llm.prompts import SYSTEM_PROMPT, build_recheck_prompt
from .pdf_io import cap_resolution, load_as_pages
from .preprocess_v3 import run_default as run_default_v3
from .schema import BBox, Cell, DocumentResult, FieldType, PageResult


# ---------------------------------------------------------------------------
# JSONパースヘルパ
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _parse_json_loose(text: str) -> dict:
    text = text.strip()
    m = _CODE_FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK_RE.search(text)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"LLM応答からJSONを抽出できませんでした: {text[:200]}...")


# ---------------------------------------------------------------------------
# 数値正規化 (バグ修正版)
# ---------------------------------------------------------------------------

_NUM_KEEP_RE = re.compile(r"[^\d\-.]")

# 全角→半角の対応表 (左右で必ず文字数を一致させる)
_ZH_TO_HALF = str.maketrans(
    "０１２３４５６７８９，．△▲−ー",  # 16文字
    "0123456789,.----",                # 16文字
)


def _normalize_number(s: str) -> Optional[float]:
    """'1,234,567' → 1234567.0, '△500' → -500.0 等を試みる."""
    if s is None:
        return None
    raw = s.strip()
    if not raw:
        return None
    raw = raw.translate(_ZH_TO_HALF)
    negative = False
    if raw.startswith(("-", "△", "▲")):
        negative = True
        raw = raw.lstrip("-△▲ ")
    if raw.startswith("(") and raw.endswith(")"):
        negative = True
        raw = raw.strip("()")
    raw = _NUM_KEEP_RE.sub("", raw)
    if raw in {"", ".", "-"}:
        return None
    try:
        val = float(raw)
        return -val if negative else val
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    provider: str = "claude"
    model: Optional[str] = None
    dpi: int = 220
    max_side: int = 1600
    do_preprocess: bool = True
    do_warp: bool = True
    do_flatten: bool = True
    recheck_threshold: float = 0.6
    max_rechecks_per_page: int = 6


class PipelineV2:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.llm = get_client(self.config.provider, self.config.model)

    def run(self, src: str | Path) -> DocumentResult:
        src = Path(src)
        t0 = time.time()
        pages = load_as_pages(src, dpi=self.config.dpi)

        page_results: list[PageResult] = []
        total_cost = 0.0
        for i, raw_img in enumerate(pages, start=1):
            img = raw_img
            if self.config.do_preprocess:
                img, _ = run_default_v3(
                    img,
                    do_warp=self.config.do_warp,
                    do_flatten=self.config.do_flatten,
                )
            img = cap_resolution(img, max_side=self.config.max_side)
            page_res, cost = self.extract_page(img, page_index=i)
            page_results.append(page_res)
            if cost:
                total_cost += cost

        return DocumentResult(
            source_path=str(src),
            num_pages=len(pages),
            pages=page_results,
            llm_model=getattr(self.llm, "_model", self.config.provider),
            elapsed_sec=round(time.time() - t0, 2),
            cost_usd=round(total_cost, 4) if total_cost else None,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
    def _llm_extract(self, image: Image.Image, prompt: str) -> LLMResponse:
        return self.llm.extract(image, prompt, system=SYSTEM_PROMPT, max_tokens=8000)

    def extract_page(self, img: Image.Image, page_index: int):
        prompt = build_extraction_prompt()
        resp = self._llm_extract(img, prompt)
        try:
            data = _parse_json_loose(resp.text)
        except Exception as e:
            return (
                PageResult(
                    page=page_index,
                    image_width=img.width,
                    image_height=img.height,
                    warnings=[f"JSONパース失敗: {e}"],
                    raw_text=resp.text[:2000],
                ),
                resp.cost_usd or 0,
            )

        cells_raw = data.get("cells", []) or []
        cells: list[Cell] = []
        for c in cells_raw:
            try:
                bbox = BBox(**c["bbox"]) if c.get("bbox") else None
                ftype = c.get("field_type", "other") or "other"
                try:
                    ftype_enum = FieldType(ftype)
                except ValueError:
                    ftype_enum = FieldType.OTHER
                value = str(c.get("value", "")).strip()
                cells.append(Cell(
                    label=c.get("label"),
                    value=value,
                    field_type=ftype_enum,
                    bbox=bbox,
                    confidence=float(c.get("confidence", 1.0)),
                    page=page_index,
                    row=c.get("row"),
                    col=c.get("col"),
                    note=c.get("note"),
                    normalized_value=(
                        _normalize_number(value)
                        if ftype_enum in {FieldType.NUMBER, FieldType.TOTAL}
                        else None
                    ),
                ))
            except Exception as e:  # noqa: BLE001
                cells.append(Cell(
                    page=page_index,
                    value=str(c.get("value", "")),
                    note=f"cell parse error: {e}",
                    confidence=0.0,
                ))

        page_res = PageResult(
            page=page_index,
            image_width=img.width,
            image_height=img.height,
            document_type=data.get("document_type"),
            title=data.get("title"),
            cells=cells,
            raw_text=data.get("raw_text"),
            warnings=list(data.get("warnings") or []),
        )

        cost = resp.cost_usd or 0
        cost += self._recheck_low_confidence(img, page_res)
        return page_res, cost

    def _recheck_low_confidence(self, img: Image.Image, page: PageResult) -> float:
        targets = [
            c for c in page.cells
            if c.confidence < self.config.recheck_threshold
            and c.bbox is not None
            and c.field_type in {FieldType.NUMBER, FieldType.TOTAL, FieldType.LABEL}
        ][: self.config.max_rechecks_per_page]
        if not targets:
            return 0.0

        cost = 0.0
        for cell in targets:
            try:
                crop = self._crop_with_padding(img, cell.bbox, pad_ratio=0.05)
                prompt = build_recheck_prompt(cell.label or "(no label)", cell.value)
                resp = self._llm_extract(crop, prompt)
                cost += resp.cost_usd or 0
                try:
                    data = _parse_json_loose(resp.text)
                except Exception:
                    continue
                new_value = str(data.get("value", "")).strip()
                if new_value and new_value != cell.value:
                    cell.note = (
                        (cell.note + " | " if cell.note else "")
                        + f"recheck: '{cell.value}' → '{new_value}'"
                    )
                    cell.value = new_value
                    if cell.field_type in {FieldType.NUMBER, FieldType.TOTAL}:
                        cell.normalized_value = _normalize_number(new_value)
                cell.confidence = max(cell.confidence, float(data.get("confidence", cell.confidence)))
            except Exception:
                continue
        return cost

    @staticmethod
    def _crop_with_padding(img: Image.Image, bbox: BBox, pad_ratio: float = 0.05) -> Image.Image:
        x, y, w, h = bbox.to_pixels(img.width, img.height)
        pad_x = int(img.width * pad_ratio)
        pad_y = int(img.height * pad_ratio)
        return img.crop((
            max(0, x - pad_x), max(0, y - pad_y),
            min(img.width, x + w + pad_x), min(img.height, y + h + pad_y),
        ))

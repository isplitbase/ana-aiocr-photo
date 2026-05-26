"""kintou_v2: 行ずれ対策強化版.

prompts_kintou_v2 (【】小計+y座標厳密対応+均等割) を使う.
label_bbox / value_bbox を保持可能.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from .llm import get_client
from .llm.base import LLMResponse
from .llm.prompts_kintou_v2 import (
    SYSTEM_PROMPT_KINTOU_V2,
    build_extraction_prompt_kintou_v2,
    build_recheck_prompt_kintou_v2,
)
from .pdf_io import cap_resolution, load_as_pages
from .pipeline_v2 import PipelineConfig, _parse_json_loose, _normalize_number
from .preprocess_v3 import run_default as run_v3
from .schema import BBox, Cell, DocumentResult, FieldType, PageResult


def _make_bbox(b: dict | None) -> Optional[BBox]:
    if not b:
        return None
    b = dict(b)
    for k in ("x", "y", "w", "h"):
        if k in b:
            b[k] = max(0, min(b[k], 1000))
    if b.get("w", 0) > 0 and b.get("h", 0) > 0:
        return BBox(**{k: b[k] for k in ("x", "y", "w", "h")})
    return None


class PipelineKintouV2:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig(
            provider="claude",
            recheck_threshold=0.85,
            max_rechecks_per_page=10,
        )
        self.llm = get_client(self.config.provider, self.config.model)

    def run(self, src: str | Path) -> DocumentResult:
        src = Path(src)
        t0 = time.time()
        pages = load_as_pages(src, dpi=self.config.dpi)
        page_results = []
        total_cost = 0.0
        for i, raw_img in enumerate(pages, start=1):
            img = raw_img
            if self.config.do_preprocess:
                img, _ = run_v3(
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
        return self.llm.extract(image, prompt, system=SYSTEM_PROMPT_KINTOU_V2, max_tokens=8000)

    def extract_page(self, img: Image.Image, page_index: int):
        prompt = build_extraction_prompt_kintou_v2()
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
        cells = []
        for c in cells_raw:
            try:
                # label_bbox / value_bbox があれば優先, なければ bbox を使う
                label_bbox = _make_bbox(c.get("label_bbox"))
                value_bbox = _make_bbox(c.get("value_bbox"))
                bbox = _make_bbox(c.get("bbox")) or value_bbox or label_bbox
                ftype = c.get("field_type", "other") or "other"
                try:
                    ftype_enum = FieldType(ftype)
                except ValueError:
                    ftype_enum = FieldType.OTHER
                value = str(c.get("value", "")).strip()
                # y座標差をチェック (label_bbox と value_bbox の両方ある場合のみ)
                y_warning = None
                if label_bbox and value_bbox:
                    dy = abs(label_bbox.y - value_bbox.y)
                    if dy > 25:  # 2.5% 以上ずれていたら警告
                        y_warning = f"label_y={label_bbox.y:.0f} value_y={value_bbox.y:.0f} dy={dy:.0f}"
                note = c.get("note") or ""
                if y_warning:
                    note = (note + " | " if note else "") + f"y-mismatch: {y_warning}"
                cells.append(Cell(
                    label=c.get("label"),
                    value=value,
                    field_type=ftype_enum,
                    bbox=bbox,
                    confidence=float(c.get("confidence", 1.0)),
                    page=page_index,
                    row=c.get("row"),
                    col=c.get("col"),
                    note=note,
                    normalized_value=(
                        _normalize_number(value)
                        if ftype_enum in {FieldType.NUMBER, FieldType.TOTAL}
                        else None
                    ),
                ))
            except Exception as e:
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
        ][: self.config.max_rechecks_per_page]
        if not targets:
            return 0.0
        cost = 0.0
        for cell in targets:
            try:
                crop = self._crop_with_padding(img, cell.bbox, pad_ratio=0.10)
                prompt = build_recheck_prompt_kintou_v2(cell.label or "(no label)", cell.value)
                resp = self._llm_extract(crop, prompt)
                cost += resp.cost_usd or 0
                try:
                    data = _parse_json_loose(resp.text)
                except Exception:
                    continue
                new_value = str(data.get("value", "")).strip()
                if new_value and new_value != cell.value:
                    if cell.field_type in {FieldType.NUMBER, FieldType.TOTAL}:
                        new_normalized = _normalize_number(new_value)
                        if new_normalized is None:
                            cell.note = (
                                (cell.note + " | " if cell.note else "")
                                + f"recheck rejected non-numeric: '{new_value}'"
                            )
                            continue
                        old_value = cell.value
                        cell.value = new_value
                        cell.normalized_value = new_normalized
                        cell.note = (
                            (cell.note + " | " if cell.note else "")
                            + f"recheck: '{old_value}' -> '{new_value}'"
                        )
                    else:
                        old_value = cell.value
                        cell.value = new_value
                        cell.note = (
                            (cell.note + " | " if cell.note else "")
                            + f"recheck: '{old_value}' -> '{new_value}'"
                        )
                cell.confidence = max(cell.confidence, float(data.get("confidence", cell.confidence)))
            except Exception:
                continue
        return cost

    @staticmethod
    def _crop_with_padding(img: Image.Image, bbox: BBox, pad_ratio: float = 0.10) -> Image.Image:
        x, y, w, h = bbox.to_pixels(img.width, img.height)
        pad_x = int(img.width * pad_ratio)
        pad_y = int(img.height * pad_ratio)
        return img.crop((
            max(0, x - pad_x), max(0, y - pad_y),
            min(img.width, x + w + pad_x), min(img.height, y + h + pad_y),
        ))

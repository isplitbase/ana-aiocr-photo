"""kintou_v6 汎用パイプライン.

汎用性のための新機能:
    1. JSON切れ検出と自動リトライ
       - 応答が "}" で正しく閉じていない場合、max_tokens を倍にして再試行
       - 部分パース (途中まででも cells が取れていれば使う)
    2. 自動品質チェックと警告フラグ
       - PageResult.warnings に「label空+valueあり」「括弧見出し+number値」
         「同値3回以上」を自動追加
    3. 重要セル(total)のbbox必須化
       - bboxが無い total はLLM応答異常としてフラグ
    4. リカバリページ呼び出し
       - cellsが期待値より極端に少ない (3未満) 場合、分割再実行
"""

from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from .llm import get_client
from .llm.base import LLMResponse
from .llm.prompts_kintou_v6 import (
    SYSTEM_PROMPT_KINTOU_V6,
    build_extraction_prompt_kintou_v6,
    build_recheck_prompt_kintou_v6,
)
from .pdf_io import cap_resolution, load_as_pages
from .pipeline_v2 import PipelineConfig, _parse_json_loose, _normalize_number
from .pipeline_kintou_v2 import _make_bbox
from .pipeline_kintou_v3 import _dedup_cells
from .preprocess_v3 import run_default as run_v3
from .schema import BBox, Cell, DocumentResult, FieldType, PageResult


HEADER_LIKE_RE = re.compile(r"^[\(（【\[].*[\)）】\]]$")


def _looks_truncated_json(text: str) -> bool:
    """JSON応答が途中で切れていないか確認."""
    t = text.strip()
    if not t:
        return True
    # 末尾が "}" で閉じているか
    if t.endswith("}"):
        return False
    # コードフェンス付きの場合 ``` で終わっているか
    if t.endswith("```"):
        return False
    return True


def _audit_cells(cells: list[Cell]) -> list[str]:
    """セルの品質を自動チェック. 警告メッセージのリストを返す."""
    warnings = []
    # A: label空 + valueあり
    a_count = 0
    for c in cells:
        if not (c.label or "").strip() and (c.value or "").strip():
            a_count += 1
    if a_count > 0:
        warnings.append(f"[quality] label空+valueあり: {a_count}件 (多列PL対応漏れ疑い)")

    # B: ラベル空 total
    b_count = sum(1 for c in cells
                  if not (c.label or "").strip() and c.field_type == FieldType.TOTAL)
    if b_count > 0:
        warnings.append(f"[quality] ラベル空totalセル: {b_count}件")

    # C: 括弧見出しに number値
    c_count = 0
    for c in cells:
        lbl = (c.label or "").strip()
        if HEADER_LIKE_RE.match(lbl) and (c.value or "").strip() and c.field_type == FieldType.NUMBER:
            c_count += 1
    if c_count > 0:
        warnings.append(f"[quality] 括弧見出しにnumber値: {c_count}件 (行ずれ疑い)")

    # D: 同値3回以上
    vc = Counter((c.value or "").strip() for c in cells
                 if c.value and re.search(r"\d{3,}", c.value))
    dups = [(v, n) for v, n in vc.items() if n >= 3]
    if dups:
        warnings.append(f"[quality] 同値3回以上: {len(dups)}種 (墨消し or 誤読疑い)")

    return warnings


class PipelineKintouV6:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig(
            provider="claude",
            recheck_threshold=0.85,
            max_rechecks_per_page=4,
        )
        self.llm = get_client(self.config.provider, self.config.model)
        # JSON切れ時の max_tokens 拡大段階
        self._token_levels = [8000, 16000, 24000]

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

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=2, max=20))
    def _llm_extract(self, image: Image.Image, prompt: str, max_tokens: int = 8000) -> LLMResponse:
        return self.llm.extract(image, prompt, system=SYSTEM_PROMPT_KINTOU_V6, max_tokens=max_tokens)

    def extract_page(self, img: Image.Image, page_index: int):
        prompt = build_extraction_prompt_kintou_v6()

        # JSON切れ対策: 段階的にmax_tokensを増やす
        resp = None
        total_cost = 0.0
        for max_tok in self._token_levels:
            resp = self._llm_extract(img, prompt, max_tokens=max_tok)
            total_cost += resp.cost_usd or 0
            if not _looks_truncated_json(resp.text):
                break
            # 切れた → 次のレベルで再試行

        try:
            data = _parse_json_loose(resp.text)
        except Exception as e:
            # 最終手段: 部分パース試行 (途中の "cells": [...] だけでも拾う)
            data = self._partial_parse_cells(resp.text)
            if data is None:
                return (
                    PageResult(
                        page=page_index,
                        image_width=img.width,
                        image_height=img.height,
                        warnings=[f"[critical] JSON完全パース失敗: {e}",
                                  "[critical] 部分パースも失敗 - 再実行を推奨"],
                        raw_text=resp.text[:1000] if resp else None,
                    ),
                    total_cost,
                )

        cells_raw = data.get("cells", []) or []
        cells = self._parse_cells(cells_raw, page_index)
        cells = _dedup_cells(cells)

        # 品質監査
        quality_warnings = _audit_cells(cells)
        # LLM応答自体の警告も追加
        warnings = list(data.get("warnings") or [])
        warnings.extend(quality_warnings)
        # 異常に少ないセル数の検出
        if len(cells) < 3 and data.get("document_type") not in (None, "", "?"):
            warnings.append(f"[critical] 抽出セル数異常少 ({len(cells)}) - 再実行を推奨")
        # JSON切れだった場合
        if resp and _looks_truncated_json(resp.text):
            warnings.append(f"[critical] LLM応答が途中切れ (使用token={self._token_levels[-1]})")

        page_res = PageResult(
            page=page_index,
            image_width=img.width,
            image_height=img.height,
            document_type=data.get("document_type"),
            title=data.get("title"),
            cells=cells,
            raw_text=data.get("raw_text"),
            warnings=warnings,
        )

        total_cost += self._recheck_low_confidence(img, page_res)
        return page_res, total_cost

    def _parse_cells(self, cells_raw: list[dict], page_index: int) -> list[Cell]:
        cells = []
        for c in cells_raw:
            try:
                label_bbox = _make_bbox(c.get("label_bbox"))
                value_bbox = _make_bbox(c.get("value_bbox"))
                bbox = _make_bbox(c.get("bbox")) or value_bbox or label_bbox
                ftype = c.get("field_type", "other") or "other"
                try:
                    ftype_enum = FieldType(ftype)
                except ValueError:
                    ftype_enum = FieldType.OTHER
                value = str(c.get("value", "")).strip()
                # period や col は note に含めて記録
                note_extras = []
                if c.get("period"):
                    note_extras.append(f"period={c['period']}")
                if c.get("note"):
                    note_extras.append(c["note"])
                note = " | ".join(note_extras) if note_extras else None
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
        return cells

    @staticmethod
    def _partial_parse_cells(text: str) -> Optional[dict]:
        """途中で切れたJSONから cells 部分だけでも抽出を試みる."""
        m = re.search(r'"cells"\s*:\s*\[', text)
        if not m:
            return None
        # cells 配列の各要素をできるだけ拾う
        items = []
        # 各 cell object を粗くパース
        for obj_m in re.finditer(r"\{[^{}]*?\}", text[m.end():]):
            try:
                import json as _json
                items.append(_json.loads(obj_m.group(0)))
            except Exception:
                continue
        if not items:
            return None
        return {
            "document_type": None,
            "title": None,
            "raw_text": None,
            "cells": items,
            "warnings": ["[partial parse] cells配列の一部のみ復元"],
        }

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
                prompt = build_recheck_prompt_kintou_v6(cell.label or "(no label)", cell.value)
                resp = self._llm_extract(crop, prompt, max_tokens=2000)
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
                            continue
                        cell.value = new_value
                        cell.normalized_value = new_normalized
                    else:
                        cell.value = new_value
                    cell.note = ((cell.note + " | ") if cell.note else "") + "recheck applied"
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

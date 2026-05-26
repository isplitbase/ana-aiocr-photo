"""v6/v7 → Analygent JSON 変換アダプタ (final6: 確定仕様 + 負債合計分類補正).

final5 + 負債合計の分類補正:
  「負債合計」「負債の部合計」「負債の部計」が誤って「負債純資産合計」分類に
  なる問題を後処理で補正 (final3のCATEGORY_MAP同期問題を確実に回避)。

担当者確定仕様 (2026-05-19):
  Q1: {"list": [...]} 形式
  Q2: type = BS/PL/販売費/製造原価
  Q3: 金額空欄ケース別 (a:列なし="" / b:判読不能=0 / c:0明記=0)
"""
from __future__ import annotations

import re
from typing import Optional

from src.schema_adapter_final3 import (
    convert_v6_to_analygent as _convert_final3,
    merge_multi_file as _merge_final3,
    _PERIODS,
    _strip,
)

_SECTIONS = ["BS", "PL", "販売費", "製造原価"]

# 売上原価セクションの構成項目ラベル
_COGS_LABEL_RE = re.compile(
    r"期首棚卸|期末棚卸|期首商品棚卸|期末商品棚卸|期首製品棚卸|期末製品棚卸|"
    r"商品仕入|当期商品仕入|仕入高|期首原材料|期末原材料"
)

# 「負債合計」系 (純資産/株主資本を含まない負債の合計行)
_LIABILITY_TOTAL_RE = re.compile(r"^負債(の部)?合計$|^負債の部計$")


def _fix_cogs_category(item: dict) -> dict:
    """PL内で売上原価構成項目が販管費に誤フォールバックされたのを補正 (破壊的)."""
    lbl = _strip(item.get("勘定科目", ""))
    if item.get("分類") == "販売費及び一般管理費" and _COGS_LABEL_RE.search(lbl):
        item["分類"] = "売上原価"
    return item


def _fix_liability_category(item: dict) -> dict:
    """「負債合計」が「負債純資産合計」に誤分類されたのを補正 (破壊的).

    「負債合計」「負債の部合計」「負債の部計」 → 分類「負債合計」
    「負債及び純資産合計」「負債・純資産の部合計」 → 分類「負債純資産合計」 (維持)
    """
    lbl = _strip(item.get("勘定科目", ""))
    if item.get("分類") == "負債純資産合計" and _LIABILITY_TOTAL_RE.match(lbl):
        item["分類"] = "負債合計"
    return item


def _apply_amount_cases(item: dict) -> dict:
    """Q3ケース別の金額補正 (破壊的)."""
    for p in _PERIODS:
        cell = item.get(p)
        if not isinstance(cell, dict):
            continue
        amt = cell.get("金額", "")
        pno = cell.get("page_no", "")
        if (amt == "" or amt is None) and (pno != "" and pno is not None):
            cell["金額"] = 0  # ケースb: 行はあるが判読不能
    return item


def _postprocess_item(item: dict, sec: str) -> None:
    """1レコードの後処理 (破壊的): 金額ケース別 + 分類補正."""
    _apply_amount_cases(item)
    if sec == "PL":
        _fix_cogs_category(item)
    if sec == "BS":
        _fix_liability_category(item)


def to_analygent_list(sections: dict) -> dict:
    """セクション別dict → {"list": [...]} 形式 (type付与・金額ケース別・分類補正)."""
    out = []
    for sec in _SECTIONS:
        for it in sections.get(sec, []) or []:
            _postprocess_item(it, sec)
            rec = {
                "勘定科目": it.get("勘定科目", ""),
                "今期": it.get("今期", {"金額": "", "page_no": ""}),
                "前期": it.get("前期", {"金額": "", "page_no": ""}),
                "前々期": it.get("前々期", {"金額": "", "page_no": ""}),
                "type": sec,
                "分類": it.get("分類", ""),
            }
            if it.get("_note"):
                rec["_note"] = it["_note"]
            out.append(rec)
    return {"list": out}


def convert_v6_to_analygent(page_results,
                             file_period: Optional[str] = None,
                             file_date: Optional[str] = None,
                             output_format: str = "list"):
    """v6/v7 OCR → Analygent JSON (確定仕様 + 負債合計補正)."""
    result = _convert_final3(page_results, file_period=file_period, file_date=file_date)
    meta = result.get("_meta", {})

    if output_format == "sections":
        for sec in _SECTIONS:
            for it in result.get(sec, []) or []:
                _postprocess_item(it, sec)
        return result

    listed = to_analygent_list(result)
    listed["_meta"] = meta
    return listed


def merge_multi_file(file_results, output_format: str = "list"):
    """複数ファイル統合 (確定仕様 + 負債合計補正)."""
    merged = _merge_final3(file_results)
    meta = merged.get("_meta", {})
    if output_format == "sections":
        for sec in _SECTIONS:
            for it in merged.get(sec, []) or []:
                _postprocess_item(it, sec)
        return merged
    listed = to_analygent_list(merged)
    listed["_meta"] = meta
    return listed

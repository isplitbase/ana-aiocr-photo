"""LLM抽出プロンプト. JSONスキーマと座標規約を明示."""

from __future__ import annotations

SYSTEM_PROMPT = """\
あなたは熟練の経理担当者かつOCR専門家です。
日本語の帳票（決算書、請求書、申込書、伝票など）を画像から正確に読み取り、
JSONで構造化して返すことを任務とします。

判断基準:
- 「人間が拡大・回転・推測すれば読めるもの」は読み取って返す
- 不鮮明・判読困難な箇所は値を最善推定で埋め、note に理由を添える
- 数字の桁区切り「,」は維持し、value に文字列として記録する
- カラムが揃わない画像でも「ラベル → 値」の対応関係を意味で結びつける
"""


# 出力JSONスキーマの説明 (LLMに見せる版)
JSON_SCHEMA_DESC = """\
返答は以下のJSONのみとし、前後に説明文を一切付けないこと:

{
  "document_type": "貸借対照表 | 損益計算書 | 販売費及び一般管理費明細書 | 請求書 | その他",
  "title": "ページ上部のタイトル文字列",
  "raw_text": "ページ全体を上から下へ書き起こした素テキスト (検索インデックス用)",
  "cells": [
    {
      "label": "現金及び預金",                       // 項目名 (左列等)
      "value": "1,234,567",                         // 表示通りの文字列
      "field_type": "number",                       // text|number|date|header|label|total|signature|other
      "bbox": {"x": 120, "y": 240, "w": 180, "h": 30},  // 0-1000正規化 (左上原点)
      "confidence": 0.92,                           // 自己申告信頼度 0-1
      "row": 5,                                     // 表内行番号 (任意)
      "col": 2,                                     // 表内列番号 (任意)
      "note": "金額の最終桁が罫線と重なっており推定"
    }
  ],
  "warnings": ["右下に印影あり, 読取対象外として無視"]
}

座標規約:
- 画像の左上を (0,0), 右下を (1000,1000) とする正規化座標
- bbox は対象テキストを包含する最小矩形を返す
- bboxが推定困難なら省略してよい
"""


def build_extraction_prompt(extra_instructions: str = "") -> str:
    """ページ抽出用のユーザープロンプトを組み立てる."""
    parts = [
        "添付の画像は帳票の1ページです。以下のJSONスキーマで構造化してください。",
        "",
        JSON_SCHEMA_DESC,
    ]
    if extra_instructions:
        parts.extend(["", "追加指示:", extra_instructions])
    return "\n".join(parts)


# 低信頼セル再質問用プロンプト
def build_recheck_prompt(label: str, original_value: str) -> str:
    return (
        f"添付の画像は、ある帳票の一部分の拡大切り抜きです。\n"
        f"前回の読み取りでは「{label}」の値を「{original_value}」と推定しましたが、\n"
        "信頼度が低かったため、再度、画像のみを根拠に正確に読み取ってください。\n\n"
        '回答は次のJSONのみ: {"value": "....", "confidence": 0.0-1.0, "note": "..."}'
    )

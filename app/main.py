"""zlite-aiocr-photo Cloud Run エントリポイント.

写真撮影されたPDF (photo_pdf) を Claude API でOCRし、
既存 Analygent JSON (list形式) に変換して返す。

エンドポイント:
  GET  /health             - ヘルスチェック
  POST /v1/pipeline        - メイン (PDF判定 + photo_pdfのみOCR)
  POST /v1/aiocr-photo     - エイリアス (同じ処理)
  POST /v1/classify        - 判定のみ (APIコスト0)

PHP側 (cash_ai_checkbyclaude.php / zlite-getpdfinfo.php) と同じパターンで
ID Token 認証付きで呼び出される想定。
"""
from __future__ import annotations

import traceback
from typing import Any, Dict

from fastapi import Body, FastAPI, HTTPException

from app.pipeline.runner import run_aiocr_photo, run_classify_only

app = FastAPI(title="zlite-aiocr-photo", version="1.0.0")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "zlite-aiocr-photo"}


@app.post("/v1/pipeline")
def pipeline(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """メインエンドポイント.

    リクエスト例:
      {
        "ai_case_id": 12345,
        "pdfurls": [
          "gs://zlite/u123-1.pdf",
          "gs://zlite/u123-2.pdf"
        ],
        "file_period": "今期",
        "file_date":   "2026-03-31",
        "model":       "claude-opus-4-7",    // 既定: claude-opus-4-7 (Haiku に戻す場合は明示)
        "max_side":    null,                  // 省略時はモデル別の自動値 (Haiku/Sonnet=1568, Opus=2576)
        "dpi":         200,
        "render_dpi":  300                    // 描画DPI (オーバーサンプル用, 既定 300)
      }

    レスポンス例:
      {
        "status": "success",
        "ai_case_id": 12345,
        "results": [
          {
            "pdf": "u123-1.pdf",
            "kind": "photo_pdf",
            "photo_score": 0.65,
            "routing": "claude_ai_ocr",
            "analygent": {"list": [...]},
            "records": 87,
            "cost_usd": 0.32
          },
          {
            "pdf": "u123-2.pdf",
            "kind": "text_pdf",
            "photo_score": 0.12,
            "routing": "existing_analygent_engine"
          }
        ],
        "total_cost_usd": 0.32
      }
    """
    try:
        return run_aiocr_photo(payload)
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print("=== /v1/pipeline ERROR START ===", flush=True)
        print(tb, flush=True)
        print("payload =", payload, flush=True)
        print("=== /v1/pipeline ERROR END ===", flush=True)
        raise HTTPException(
            status_code=500,
            detail={
                "message": str(e),
                "traceback_tail": tb.splitlines()[-20:],
            },
        )


@app.post("/v1/aiocr-photo")
def aiocr_photo(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """/v1/pipeline と同じ処理 (専用名エンドポイント)."""
    return pipeline(payload)


@app.post("/v1/classify")
def classify(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """PDF判定のみ実行 (APIコスト0). 振り分け結果だけ返す."""
    try:
        return run_classify_only(payload)
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print("=== /v1/classify ERROR ===", flush=True)
        print(tb, flush=True)
        raise HTTPException(
            status_code=500,
            detail={
                "message": str(e),
                "traceback_tail": tb.splitlines()[-20:],
            },
        )

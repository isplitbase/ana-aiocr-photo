"""出力スキーマ定義.

LLMが返すJSONをPydanticでバリデーション・正規化し、
JSON / CSV / Excel へエクスポートする際の共通モデルとする.

座標系の方針:
    LLMには「画像左上を(0,0)とし、右下を(1000,1000)とする正規化座標」で
    bboxを返してもらう (Claude/Geminiともこの形式が安定).
    実画像解像度へのスケーリングはパイプライン側で行う.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class BBox(BaseModel):
    """正規化バウンディングボックス (0-1000).

    左上原点, 右下が(1000,1000). 画像アスペクト比に依らず使える.
    """

    x: float = Field(..., ge=0, le=1000, description="左上x (0-1000)")
    y: float = Field(..., ge=0, le=1000, description="左上y (0-1000)")
    w: float = Field(..., gt=0, le=1000, description="幅 (0-1000)")
    h: float = Field(..., gt=0, le=1000, description="高さ (0-1000)")

    def to_pixels(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        """実画像ピクセル座標(x, y, w, h)に変換."""
        return (
            int(self.x / 1000 * img_w),
            int(self.y / 1000 * img_h),
            int(self.w / 1000 * img_w),
            int(self.h / 1000 * img_h),
        )


class FieldType(str, Enum):
    """項目の種別."""

    TEXT = "text"          # 一般的なテキスト
    NUMBER = "number"      # 数値 (金額・数量)
    DATE = "date"          # 日付
    HEADER = "header"      # 表のヘッダ・タイトル
    LABEL = "label"        # 勘定科目名などラベル
    TOTAL = "total"        # 合計行
    SIGNATURE = "signature"  # 印影・署名
    OTHER = "other"


class Cell(BaseModel):
    """抽出された1セル分のデータ."""

    label: Optional[str] = Field(None, description="項目名 (例: '現金及び預金')")
    value: str = Field(..., description="生のテキスト値 (例: '1,234,567')")
    normalized_value: Optional[float] = Field(
        None, description="数値正規化後の値 (NUMBER/TOTALのみ)"
    )
    field_type: FieldType = Field(FieldType.OTHER)
    bbox: Optional[BBox] = Field(None, description="正規化bbox")
    confidence: float = Field(
        1.0, ge=0, le=1, description="LLMの自己申告信頼度 (0-1)"
    )
    page: int = Field(1, ge=1, description="所属ページ番号 (1-indexed)")
    row: Optional[int] = Field(None, description="表内の行番号 (任意)")
    col: Optional[int] = Field(None, description="表内の列番号 (任意)")
    note: Optional[str] = Field(None, description="LLMが付した補足 (読みにくい等)")

    @field_validator("value")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class PageResult(BaseModel):
    """1ページ分の抽出結果."""

    page: int
    image_width: int
    image_height: int
    document_type: Optional[str] = Field(
        None, description="LLMが判定した書類種別 (例: '貸借対照表')"
    )
    title: Optional[str] = None
    cells: List[Cell] = Field(default_factory=list)
    raw_text: Optional[str] = Field(
        None, description="LLMが抽出したページ全体の素テキスト (検索用)"
    )
    warnings: List[str] = Field(default_factory=list)


class DocumentResult(BaseModel):
    """ドキュメント全体の抽出結果."""

    source_path: str
    num_pages: int
    pages: List[PageResult]
    llm_model: str
    elapsed_sec: float
    cost_usd: Optional[float] = None

    def all_cells(self) -> List[Cell]:
        return [c for p in self.pages for c in p.cells]

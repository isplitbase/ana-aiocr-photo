"""kintou_v7: v6 + col独立ラベル取得プロンプト.

PipelineKintouV6 を継承し、プロンプトだけ v7 に差し替えた最小実装。
"""
from __future__ import annotations

from .pipeline_kintou_v6 import PipelineKintouV6
from .llm.prompts_kintou_v7 import (
    SYSTEM_PROMPT_KINTOU_V7,
    build_extraction_prompt_kintou_v7,
    build_recheck_prompt_kintou_v7,
)


class PipelineKintouV7(PipelineKintouV6):
    """v6パイプラインのプロンプトをv7に差し替え."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # プロンプト系をv7に差し替え
        # ※ pipeline_kintou_v6 内で SYSTEM_PROMPT_KINTOU_V6 等を直接importしてる箇所は
        #   モンキーパッチで上書きが必要
        import src.pipeline_kintou_v6 as v6mod
        v6mod.SYSTEM_PROMPT_KINTOU_V6 = SYSTEM_PROMPT_KINTOU_V7
        v6mod.build_extraction_prompt_kintou_v6 = build_extraction_prompt_kintou_v7
        v6mod.build_recheck_prompt_kintou_v6 = build_recheck_prompt_kintou_v7

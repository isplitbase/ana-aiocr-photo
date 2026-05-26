FROM python:3.12-slim

WORKDIR /app

# 必要なネイティブ依存:
#   - poppler-utils : pdf2image (PDF→画像変換) で使用
#   - libgl1 / libglib2.0-0 / libsm6 / libxext6 / libxrender1 : opencv-python(cv2) 実行時依存
#   - ghostscript : PDF処理のフォールバック
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        poppler-utils \
        ghostscript \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# 依存ライブラリ
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# アプリ本体
COPY . /app

# Cloud Run は 8080 固定
ENV PORT=8080
EXPOSE 8080

# AI-OCRの内部import (from src.pdf_classifier_v3 import ...) が解決できるよう
# app/pipeline/ を PYTHONPATH に追加
ENV PYTHONPATH=/app:/app/app/pipeline

# 起動
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

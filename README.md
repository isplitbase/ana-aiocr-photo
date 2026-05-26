# zlite-aiocr-photo

写真撮影された決算書PDF (photo_pdf) を Claude API で OCR し、既存 Analygent の JSON 形式 (`{"list":[...]}`) に変換して返す Cloud Run サービス。

判定で `text_pdf` / `scan_pdf` と認識されたものは既存 Analygent (Gemini/GPT) で処理すべきと返却するだけで、本サービスは API を叩かない。

---

## アーキテクチャ

```
[PHP] zlite-getpdfinfo.php / do_ai.php
   │
   ▼ POST + ID Token
[Cloud Run] zlite-aiocr-photo  /v1/pipeline
   │
   ├─ GCSからPDFをダウンロード (google-cloud-storage)
   ├─ pdf_classifier_v3 で photo / text / scan を判定
   ├─ photo_pdf のみ pipeline_kintou_v7 で Claude OCR
   └─ schema_adapter_final6 で Analygent JSON (list形式) に変換
```

OCRエンジン本体 (`app/pipeline/src/`) は `AI-OCR_*.zip` のソースを未改変で配置しています。差し替え時は `src/` を上書きするだけで OK。

---

## エンドポイント

### `GET /health`
ヘルスチェック。`{"ok": true}` を返す。

### `POST /v1/pipeline`
メインエンドポイント。

**リクエスト**
```json
{
  "ai_case_id": 12345,
  "pdfurls": [
    "gs://zlite/u123-1.pdf",
    "gs://zlite/u123-2.pdf"
  ],
  "file_period": "今期",
  "file_date":   "2026-03-31",
  "model":       "claude-haiku-4-5-20251001",
  "max_side":    1400,
  "dpi":         200,
  "classify_only": false,
  "nodoai":      false
}
```

| キー | 型 | 必須 | 説明 |
|---|---|---|---|
| `ai_case_id` | int | 任意 | レスポンスにそのまま返す |
| `pdfurls` (or `files` / `file`) | string[] | ★ | GCS URI のリスト (`gs://bucket/key`) |
| `file_period` | string | 任意 | `今期` / `前期` / `前々期` |
| `file_date` | string | 任意 | `YYYY-MM-DD` |
| `model` | string | 任意 | デフォルト `claude-haiku-4-5-20251001` |
| `max_side` | int | 任意 | 画像長辺ピクセル (デフォルト 1400) |
| `dpi` | int | 任意 | レンダリングDPI (デフォルト 200) |
| `classify_only` | bool | 任意 | true なら判定のみで Claude を呼ばない |
| `nodoai` | bool | 任意 | true なら処理スキップ (互換) |

**レスポンス (成功)**
```json
{
  "status": "success",
  "ai_case_id": 12345,
  "results": [
    {
      "pdf": "u123-1.pdf",
      "gs_uri": "gs://zlite/u123-1.pdf",
      "kind": "photo_pdf",
      "photo_score": 0.65,
      "num_pages": 3,
      "routing": "claude_ai_ocr",
      "analygent": { "list": [ /* Analygent 確定仕様 */ ] },
      "records": 87,
      "cost_usd": 0.32,
      "elapsed_sec": 41.2
    },
    {
      "pdf": "u123-2.pdf",
      "kind": "text_pdf",
      "photo_score": 0.12,
      "routing": "existing_analygent_engine",
      "note": "現行Analygent (Gemini/GPT) で処理すべき"
    }
  ],
  "routing_counts": { "claude_ai_ocr": 1, "existing_analygent_engine": 1 },
  "total_cost_usd": 0.32,
  "model": "claude-haiku-4-5-20251001",
  "created_at": "2026-05-25T00:00:00Z"
}
```

### `POST /v1/aiocr-photo`
`/v1/pipeline` のエイリアス (同じ処理)。

### `POST /v1/classify`
判定のみ実行。Claude API は呼ばないので無料で振り分け確認できる。

---

## 環境変数

| 変数 | 必須 | 説明 |
|---|---|---|
| `ANTHROPIC_API_KEY` | ★ | Claude API キー。**Cloud Run の環境変数 / Secret Manager で必ず注入する**。未設定で `/v1/pipeline` を呼ぶと 500 (RuntimeError) を返す。`/v1/classify` (判定のみ) は不要 |
| `CLAUDE_MODEL` | 任意 | リクエストに `model` が無い場合のデフォルト |
| `GCP_PROJECT` (or `GOOGLE_CLOUD_PROJECT`) | 任意 | GCS クライアントの project を明示したい場合のみ。Cloud Run ではメタデータサーバから自動取得されるので通常不要 |
| `GOOGLE_APPLICATION_CREDENTIALS` | 任意 | **ローカル開発時のみ**。サービスアカウント JSON のパスを指定。Cloud Run 上ではランタイムのサービスアカウントが自動使用される (ADC) |
| `PORT` | - | Cloud Run が自動付与 (8080) |

### GCS 認証

Cloud Run のランタイムサービスアカウントに対象バケットの `roles/storage.objectViewer` 以上の権限を付与しておく。
コード側は ADC (Application Default Credentials) を使うため、明示的なキー管理は不要。

ローカル開発の場合:
```bash
# 方法1: gcloud のユーザー認証を流用
gcloud auth application-default login

# 方法2: サービスアカウント JSON を直接指定
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
```

### Anthropic API キーの取扱

ソース埋め込みは廃止しました。Cloud Run の環境変数として `ANTHROPIC_API_KEY` を必ず設定してください。

設定方法は2通り、どちらでもコード側の挙動は同じ:

**方法A: 普通の環境変数 (シンプル)**

GCP Console → Cloud Run → サービス → 「変数とシークレット」タブで `ANTHROPIC_API_KEY` を追加するか、CLI で:
```bash
gcloud run services update zlite-aiocr-photo \
    --region asia-northeast1 \
    --update-env-vars ANTHROPIC_API_KEY=sk-ant-...
```
注: 値が Cloud Run の設定にそのまま保存されるので、サービスの閲覧権限を持つ人には見える。

**方法B: Secret Manager 経由 (本番推奨)**

```bash
# 初回 (シークレット作成)
gcloud secrets create anthropic-api-key --replication-policy=automatic
echo -n "sk-ant-..." | gcloud secrets versions add anthropic-api-key --data-file=-

# Cloud Run のサービスアカウントに roles/secretmanager.secretAccessor を付与
gcloud secrets add-iam-policy-binding anthropic-api-key \
    --member="serviceAccount:SERVICE_ACCOUNT_EMAIL" \
    --role="roles/secretmanager.secretAccessor"
```

利点: 値が Cloud Run の設定画面に出ない / ローテーションは `gcloud secrets versions add` で新値を投入するだけ (`:latest` 参照なら再デプロイ不要) / Secret Manager 単位でアクセス権を別管理できる。

---

## デプロイ

Cloud Run へのデプロイは、Dockerfile を含む本ディレクトリのルートを対象にビルド。

```bash
# 例A: 普通の環境変数で API キーを渡す (シンプル)
gcloud run deploy zlite-aiocr-photo \
    --source . \
    --region asia-northeast1 \
    --no-allow-unauthenticated \
    --memory 4Gi --cpu 2 --timeout 1800 \
    --set-env-vars CLAUDE_MODEL=claude-haiku-4-5-20251001,ANTHROPIC_API_KEY=sk-ant-...

# 例B: Secret Manager 経由 (本番推奨)
gcloud run deploy zlite-aiocr-photo \
    --source . \
    --region asia-northeast1 \
    --no-allow-unauthenticated \
    --memory 4Gi --cpu 2 --timeout 1800 \
    --set-env-vars CLAUDE_MODEL=claude-haiku-4-5-20251001 \
    --set-secrets ANTHROPIC_API_KEY=anthropic-api-key:latest
```

prod は別リポジトリ `zlite-aiocr-photo-real` をデプロイ。コード自体は dev と同一で、デプロイ先 (Cloud Run service 名) と環境変数だけ切り替える。

---

## PHP 側からの呼び出し

`sapis/cash_ai_checkbyclaude.php` と同じ流儀 (Service Account ID Token + curl POST) で叩く。`sapis/zlite-getpdfinfo.php` の Cloud Run 呼び出し直後に photo_pdf 分岐を追加するのが推奨ルート。

```php
// 概略
$service_url = "https://zlite-aiocr-photo-512697354748.asia-northeast1.run.app";
if ($port == "8012") {
    $service_url = "https://zlite-aiocr-photo-real-512697354748.asia-northeast1.run.app";
}
$audience = $service_url;
$url = rtrim($service_url, "/") . "/v1/pipeline";

$idToken = getCloudRunIdToken($serviceAccountJsonPath, $audience);
$payload = [
    "ai_case_id"   => $ai_case_id,
    "pdfurls"      => $gsUrls,   // ["gs://bucket/key", ...]
    "file_period"  => $file_period,
    "file_date"    => $file_date,
];
list($respBody, $errno, $err, $httpCode)
    = postJsonToCloudRun($url, $idToken, $payload);
```

---

## ディレクトリ構成

```
zlite-aiocr-photo/
├── Dockerfile
├── requirements.txt
├── README.md
├── .gitignore
└── app/
    ├── __init__.py
    ├── main.py                          # FastAPI ルータ
    └── pipeline/
        ├── __init__.py
        ├── runner.py                    # GCS DL → 分類 → OCR → Analygent 変換
        └── src/                         # AI-OCR エンジン本体 (未改変)
```

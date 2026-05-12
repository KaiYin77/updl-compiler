---
id: TKT-001
type: feature
status: todo
priority: high
created: 2026-05-11
updated: 2026-05-11
related_wiki:
depends_on:
---

# TKT-001：Speaker ID 邊緣裝置範例（訓練→量化→編譯）

## 需求描述

在 `examples/speaker_id/` 建立完整範例，示範 Text-Dependent Speaker Verification
從訓練到部署至 UP301 的全流程。參考 `examples/keyword_spotting/` 結構，
但目標從關鍵字分類改為說話者身分驗證（accept / reject）。

**核心需求（本階段：Ahead-of-Time 離線流程）：**
- 輸入音訊規格：16kHz，PDM mic 輸出
- 本階段交付：訓練好的 Speaker Embedding 模型，經量化與編譯後輸出 C 陣列
- 本階段**不含**：Enrollment 邏輯、Cosine Similarity Runtime、Embedding 寫入 Flash
  （上述三項為裝置端 On-Device 開發，下一階段處理）
- 最終輸出：UPDL compiler → INT16 UPH5 C 陣列（UP301 路徑）

## 技術方向

| 層級 | 技術選擇 |
|------|---------|
| 特徵提取 | MFCC（複用 KWSPreprocessor，10 coefficients） |
| 模型骨架 | DS-CNN（複用 KWS 架構） |
| 模型輸出頭 | Dense(128) + L2-Norm embedding（取代 KWS 的 Softmax） |
| 訓練 Loss | Softmax with speaker labels（MVP；ArcFace 有餘裕再升級） |
| 量化 | PTQ INT16（UPDL compiler；QAT 有餘裕再升級） |
| 資料集 | CN-Celeb v2 |
| 噪音增強 | MUSAN + RIRS（dataset 管線中實作） |

## 實作原則

**每個 example 完全自包含，不跨目錄 import。**
KWS 的相關檔案直接複製進 `examples/speaker_id/` 後再改，
不使用繼承或 `from examples.keyword_spotting import ...`。

## 從 KWS 複製後修改的檔案

| 來源（KWS） | 複製為（SID） | 修改幅度 |
|------------|--------------|---------|
| `kws_preprocessor.py` | `sid_preprocessor.py` | 小（只換 class 名） |
| `kws_model.py` | `sid_model.py` | 中（head 改 L2-Norm） |
| `kws_compiler.py` | `sid_compiler.py` | 小（換路徑與 prefix） |
| `kws_generate_test_input_fp32.py` | `sid_generate_test_input_fp32.py` | 中（換 dataset / label） |

## 需新增的 Gap（本階段）

1. **模型定義**：DS-CNN head 改為 `Dense(128) → L2 Norm`，移除 Softmax
2. **訓練腳本**：Softmax by speaker ID loss，輸出 embedding extractor SavedModel
3. **Dataset 管線**：`/home/kaiyin-upbeat/data` 不存在時自動從 OpenSLR 下載並解壓，再載入 CN-Celeb v2 + MUSAN 噪音增強

**不在本階段：**
- Enrollment 腳本（裝置端開發）
- Cosine similarity 推論邏輯（裝置端開發）
- Embedding 寫入 Flash（裝置端開發）

## 預期產出檔案結構

```
examples/speaker_id/
├── sid_model.py                  # DS-CNN + L2-Norm embedding head
├── sid_preprocessor.py           # 複製自 kws_preprocessor.py，只換 class 名
├── sid_dataset.py                # CN-Celeb v2 loader（資料不存在時自動下載解壓）
├── sid_train.py                  # Softmax speaker 訓練
├── sid_compiler.py               # PTQ + UPDL 編譯
├── sid_generate_test_input_fp32.py  # 測試輸入產生（FP32）
├── ref_model/                    # 訓練完成的 SavedModel
└── uph5/                         # UPDL 輸出 C 陣列
```

> Enrollment、Cosine Similarity Runtime、Flash 寫入不在此目錄，屬裝置端開發。

## 驗收條件

- [ ] `sid_train.py` 可在 CN-Celeb v2 上跑完一輪訓練並存 SavedModel
- [ ] 模型輸出為 128-dim L2-normalized embedding（不是分類 logits）
- [ ] `sid_compiler.py` 輸出合法 UPH5 C 陣列（可對照 KWS 輸出驗證格式）
- [ ] `sid_generate_test_input_fp32.py` 可產生測試用 embedding 輸入 C 陣列

## 相關檔案／模組（複製來源，不 import）

- `examples/keyword_spotting/kws_model.py` — 複製起點
- `examples/keyword_spotting/kws_preprocessor.py` — 複製起點
- `examples/keyword_spotting/kws_compiler.py` — 複製起點

## 備註

- Deadline：1-2 週，優先跑通 AoT pipeline，精度優化（ArcFace / QAT）列後續
- CN-Celeb v2 下載：https://openslr.trmal.net/resources/82/cn-celeb_v2.tar.gz → 解壓至 `/home/kaiyin-upbeat/data`
- Enrollment / Cosine Similarity / Flash 寫入屬下一階段裝置端開發，本票不追蹤

## 修改歷程

- 2026-05-11 `[建立]` 建立 ticket，需求訪問確認技術方向
- 2026-05-11 `[更新]` 明確實作原則：各 example 完全自包含，KWS 檔案直接複製不繼承
- 2026-05-11 `[更新]` 縮減範圍至 AoT 離線流程；Enrollment / Cosine Sim / Flash 寫入移至裝置端下一階段
- 2026-05-11 `[更新]` 移除 TFLite 路徑，demo stack 固定為 UPDL compiler + UPDL runtime

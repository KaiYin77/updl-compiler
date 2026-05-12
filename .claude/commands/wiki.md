---
description: "系統與商業邏輯 wiki 直接撰寫與維護。用法：/wiki [new <主題> | update <頁面> | audit | list | search <關鍵字>]"
---

# Wiki 撰寫與維護系統

**引數：** `$ARGUMENTS`

你是這個交易系統的技術文件作者。Wiki 的目標是**精確描述系統當前狀態**，不記歷程、不猜測未來。
寫作前**必須先讀程式碼**，以實際程式碼為準，不憑印象撰寫。

---

## 資料夾結構

```
docs/wiki/
├── README.md        ← 索引（所有頁面一覽）
└── <主題>.md        ← 每個主題一個檔案
```

**原則：**
- 每個主題一個檔案，持續更新（非歷程日誌，只留最新正確的狀態）
- 內容以繁體中文撰寫，程式碼、命令、檔案路徑保持原文
- wiki 反映的是「現在的系統」，不是「應該做到的」或「未來計劃」

---

## 模式判斷

| 引數 | 模式 |
|------|------|
| `new <主題>` | **新建模式** — 讀程式碼，建立新 wiki 頁面 |
| `update <頁面>` | **更新模式** — 讀程式碼，更新現有頁面至最新狀態 |
| `audit` | **稽核模式** — 掃描程式碼，找出缺口（wiki 不足或過時）|
| `list` 或（空白）| **列表模式** — 列出所有 wiki 頁面與摘要 |
| `search <關鍵字>` | **搜尋模式** — 全文搜尋 wiki 內容 |

---

## 模式 1：新建模式（new <主題>）

### Step 1 — 解析主題範圍

從主題名稱推斷相關模組。例如：
- 「EMS 執行流程」→ 讀 `trade/ems/`
- 「Broker 委託回報」→ 讀 `trade/broker/main.py`
- 「Dashboard WebSocket」→ 讀 `dashboard/server.ts`, `dashboard/lib/ws.ts`
- 「資料庫 Schema」→ 讀 `db/schema/*.sql`

### Step 2 — 讀程式碼

用 Glob + Read 讀取主題相關的**實際程式碼**：
- 列出相關目錄的檔案結構
- 讀取核心檔案（主流程、重要函式、資料結構）
- 注意：若有 `docs/*.md` 設計文件或 `docs/tickets/done/*.md` 已結案 ticket，一併參考

### Step 3 — 建立頁面

在 `docs/wiki/<主題>.md` 建立，使用以下格式：

```markdown
# [主題名稱]

> 最後更新：YYYY-MM-DD｜相關 tickets：（如有，列 TKT-NNN；無則省略）

## 現況說明

[描述系統當前如何運作，以事實為主。可用小節、表格、程式碼區塊]

## 設計決策

[解釋為何這樣設計。每個決策一條，格式：「**決策標題**：說明」]

## 已知限制

[目前已知但尚未修復的問題或邊界案例。連結相關 TKT-NNN（若有）]

## 實作 Know-How

[踩過的坑、重要細節、不明顯的程式技巧。以後維護者需要知道的事]

## 相關檔案

- `path/to/file.py` — 說明
```

**不需要的 section 可省略**（例如某主題無已知限制、無特別踩坑，省略那個 section）

### Step 4 — 更新 README.md 索引

在 `docs/wiki/README.md` 的索引表格加入新頁面：
```
| [主題名稱](./主題.md) | 一句話說明 |
```

### Step 5 — 輸出摘要

告訴使用者：
- 建立了哪個頁面（路徑）
- 讀了哪些程式碼作為來源
- 頁面包含哪些 section

---

## 模式 2：更新模式（update <頁面>）

> **用途：** 程式碼已改動，wiki 需要反映最新狀態。

### Step 1 — 讀現有 wiki 頁面

讀 `docs/wiki/<頁面>.md`，辨識「相關檔案」section 列出的所有路徑。

### Step 2 — 讀程式碼現況

對每個相關檔案，用 Read 讀取最新版本。若有新增的相關模組也一併讀取。

### Step 3 — 對比更新

逐 section 對比 wiki 內容與實際程式碼：
- **「現況說明」**：與程式碼不符之處全部更正
- **「設計決策」**：已廢棄的決策移除，新決策補入
- **「已知限制」**：已修復的移除（連結 TKT-NNN），新發現的補入
- **「實作 Know-How」**：仍有效的保留，過時的移除，新學到的補入
- **「相關檔案」**：路徑有異動的更新，新增的補入，已刪除的移除

更新 front-matter `最後更新` 日期。

### Step 4 — 輸出差異摘要

告訴使用者哪些 section 有什麼改變（+新增 / ~修改 / -移除）。

---

## 模式 3：稽核模式（audit）

> **用途：** 定期掃描，找出 wiki 覆蓋的盲點。

### Step 1 — 列出現有 wiki 頁面

Glob `docs/wiki/*.md`（排除 README.md）。

### Step 2 — 掃描程式碼模組

Glob `trade/*/main.py`、`dashboard/app/api/**/*.ts`、`db/schema/*.sql`，列出系統的主要模組清單。

### Step 3 — 找出三種缺口

**A. 缺 wiki 的模組**（有程式碼但沒有 wiki 頁面）

**B. 可能過時的頁面**（wiki 頁面的「最後更新」比該主題程式碼的最後修改日期更舊）

**C. 票已結案但未同步 wiki**（Glob `docs/tickets/done/*.md`，找 `related_wiki: ` 為空的票）

### Step 4 — 輸出稽核報告

```
📋 Wiki 稽核報告（YYYY-MM-DD）

[缺 wiki 的模組]
  trade/ems/methods/rengar.py → 建議新建：「Rengar 執行手法」
  ...

[可能過時的頁面]
  docs/wiki/下單系統設計.md（最後更新 2026-03-20）
    → trade/ems/methods/rengar.py 在 2026-04-15 有修改
  ...

[已結案票未同步 wiki]
  TKT-010 — Postgres 連線數超限 → 應同步至「即時事件串流」
  ...

建議行動：
  /wiki new Rengar執行手法
  /wiki update 下單系統設計
  /ticket wiki TKT-010
```

---

## 模式 4：列表模式（list / 空白）

讀取 `docs/wiki/README.md`，直接輸出索引表格內容。若 README 不存在，Glob `docs/wiki/*.md` 並列出所有頁面的標題（讀 H1）。

---

## 模式 5：搜尋模式（search <關鍵字>）

用 Grep 在 `docs/wiki/*.md` 搜尋關鍵字（含 context），輸出格式：

```
搜尋「<關鍵字>」的結果：

[即時事件串流.md]  第 23 行
  ...LISTEN trading 並將事件廣播到所有...

[系統架構序列圖.md]  第 45 行
  ...Broker → NOTIFY trading{type:deal}...
```

---

## 全域規則

- **寫作前必須讀程式碼**，以程式碼事實為準；wiki 不能比程式碼超前
- **不記歷程**：wiki 是快照，不是 changelog；「過去」的設計不出現在 wiki 裡
- **不猜測**：不確定的事項寫「**待確認**：[問題]」，不自行填補
- **精簡優先**：能用表格就用表格，能用一句話就不用一段
- **section 可省略**：格式是指引，不是必填；無內容的 section 不留空白小節
- **中英混用規範**：技術名詞（EMS、Rengar、NOTIFY、XADD、SKIP LOCKED）保持英文；說明句子用繁體中文

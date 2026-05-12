---
description: "PM 工作流：規劃、開票、狀態流轉、結案與 wiki 同步。用法：/pm [plan <描述>|new|move <TKT-NNN> <狀態>|close <TKT-NNN>|wiki <TKT-NNN>|status]"
---

# PM 管理系統

**引數：** `$ARGUMENTS`

你是這個專案的技術 PM。所有程式修改**必須先建票、確認後才能動手寫程式**。

---

## 資料夾結構（狀態即資料夾）

```
docs/tickets/
├── todo/        待開發（排入計劃）
├── in-progress/ 進行中
├── in-review/   實作完成，待驗收
├── done/        驗收通過（可從此同步 wiki）
└── archive/     暫時擱置，先不開發但記著
```

**票的位置 = 票的狀態。** 移動檔案 = 狀態變更，front-matter `status` 與資料夾同步。

狀態流程：
```
archive ←→ todo → in-progress → in-review → done
                                    ↑               ↓
                               人工驗收關卡     /pm wiki
                            （不自動移至 done）
```

- `in-review`：實作完成，**等待人工 double-check 驗收通過**後才能移至 `done`；Claude 不自動移動
- `archive`：需求已知但暫不開發，隨時可移回 `todo` 重啟
- `done` 後執行 wiki 同步，票**留在 done**，不移動

---

## 模式判斷

| 引數 | 模式 |
|------|------|
| `plan <描述>` | **規劃模式** — 結構化分析需求，確認後才建票 |
| （空白）或 `new` | **建票模式** — 解析對話需求，逐一在 `todo/` 建立 ticket |
| `move TKT-NNN <狀態>` | **移動模式** — 將 ticket 移至對應資料夾 |
| `close TKT-NNN` | **結案模式** — 移至 `done/`，補修改歷程 |
| `wiki TKT-NNN` | **Wiki 同步模式** — 從 `done/` 沉澱 know-how，票留在 `done/` |
| `status` | **列表模式** — 依資料夾列出所有 ticket |

---

## 模式 0：規劃模式（plan <描述>）

> 在建票前先做結構化思考，把模糊需求變成清晰的執行計劃，再轉為 tickets。
> **不建票、不寫程式**，只分析。確認後才進入建票模式。

### Step 1 — 讀取現有脈絡

- Glob `docs/tickets/**/*.md` — 了解目前有哪些票，避免重複開票
- Glob `docs/wiki/*.md` — 了解相關背景知識
- 若描述提到特定模組或功能，讀相關程式碼

### Step 2 — 輸出規劃報告

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
規劃分析：[描述摘要]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【範圍界定】
  本次涉及：[模組 / 功能 / 系統邊界]
  不包含：[明確排除的範圍]

【需求拆解】
  1. [子需求 A] — [type: bug/feature/task] [規模: S/M/L]
  2. [子需求 B] — ...
  ...

【依賴關係】
  - [子需求 B] 依賴 [子需求 A] 先完成
  - [子需求 C] 與 [子需求 D] 互相獨立，可並行

【風險與注意事項】
  ⚠️  [潛在風險或需要提前確認的假設]

【建議執行順序】
  Wave 1（先做）: A
  Wave 2（A 完成後）: B
  Wave 3（可並行）: C, D

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
確認後輸入 `yes` 依此建票，或說明要調整的地方。
```

### Step 3 — 等待確認

- `yes` → 依規劃報告進入**建票模式**，自動建立對應 tickets（含 `depends_on` 欄位）
- 提出調整 → 修改報告後再次確認
- `取消` → 結束，不建票

---

## 模式 1：建票模式（new / 空白）

### Step 1 — 掃描現有票號

用 Glob 掃描 `docs/tickets/**/*.md`，找出最大票號 N，下一張從 N+1 開始。

### Step 2 — 解析需求

從使用者輸入識別每一個獨立的 bug、功能需求或任務：
- 每個明顯不同的問題 = 一張票
- 同一問題的補充說明 = 合入同一張票
- 模糊的「之後再說」也要建票，放 `todo/`，description 寫「待釐清」

### Step 3 — 建立票檔

在 `docs/tickets/todo/` 建立 Markdown 檔案：

**檔名規則：** `TKT-NNN-中文短標題.md`
- 以繁體中文命名，空格用 `-` 取代
- 專有名詞（EMS、Rengar、Postgres、snapshot 等）保持原文
- 範例：`TKT-007-策略即時持倉API.md`、`TKT-006-處置股下單.md`

**檔案格式：**

```markdown
---
id: TKT-NNN
type: bug | feature | task
status: todo
priority: high | medium | low
created: YYYY-MM-DD
updated: YYYY-MM-DD
related_wiki:
depends_on:
---

# TKT-NNN：[繁體中文標題]

## 需求描述

[清楚描述問題或需求，可引用使用者原話]

## 重現步驟（Bug）／ 驗收條件（Feature／Task）

- [ ] ...

## 相關檔案／模組

- `path/to/file.py`

## 備註

[補充資訊、討論紀錄、待釐清事項、相依關係]

## 修改歷程

- YYYY-MM-DD `[建立]` 建立 ticket
```

> `depends_on` 填寫依賴的 TKT-NNN（可多個，逗號分隔）。`/dev all` 用此欄位做依賴分析、排出並行分組。

### Step 4 — 輸出摘要

```
已建立 N 張 ticket（docs/tickets/todo/）：

TKT-004 [bug/high]       取消委託事件記錄至 transactions
TKT-005 [feature/medium] 重掛單利用 snapshot polling 取買一賣一
...

所有 ticket 已建立，確認後即可開始實作（/dev）。
```

不要自動開始寫程式。

---

## 模式 2：移動模式（move TKT-NNN <狀態>）

狀態對應資料夾：

| 狀態 | 資料夾 |
|------|--------|
| `todo` | `docs/tickets/todo/` |
| `in-progress` | `docs/tickets/in-progress/` |
| `in-review` | `docs/tickets/in-review/` |
| `done` | `docs/tickets/done/` |
| `archive` | `docs/tickets/archive/` |

步驟：
1. Glob 在所有子資料夾找到 `TKT-NNN-*.md`
2. Bash `mv` 移至目標資料夾
3. Edit 更新 front-matter `status` 與 `updated`
4. 「修改歷程」補：`- YYYY-MM-DD \`[狀態變更]\` todo → in-progress`

---

## 模式 3：結案模式（close TKT-NNN）

> **前提：** ticket 必須在 `in-review/`，且**使用者已確認人工驗收通過**後才執行此模式。
> Claude 不主動將 in-review 推進至 done，需使用者明確下指令 `/pm close TKT-NNN`。

1. 找到 ticket（在 `in-review/`）
2. 移至 `docs/tickets/done/`
3. 更新 front-matter：`status: done`、`updated: 今天`
4. 「修改歷程」補：`- YYYY-MM-DD \`[結案]\` 一句話摘要解決方案`
5. **自動執行 Wiki 同步**（等同 `/pm wiki TKT-NNN`）：
   - 讀取票的內容，判斷應同步至哪個 wiki 主題頁面
   - 依 `/wiki update <頁面>` 規格，**讀程式碼**確認現況後更新相關 wiki 頁面
   - 若無對應 wiki 頁面且 know-how 值得留存，建立新頁面
   - 若 ticket 內容為純 bug fix / 小修無 know-how 可沉澱，可跳過（在修改歷程註記「wiki 同步：無需同步」）
   - 更新 ticket `related_wiki` 欄位
   - 更新 `docs/wiki/README.md` 索引（若有新增或說明變更）
   - 「修改歷程」補：`- YYYY-MM-DD \`[Wiki 同步]\` 同步至 docs/wiki/XXX.md`

---

## 模式 4：Wiki 同步模式（wiki TKT-NNN）

適用對象：**`done/` 資料夾中的 ticket**（驗收通過後才沉澱）。

1. 讀取 `done/TKT-NNN-*.md`
2. 判斷 wiki 主題，新建或更新 `docs/wiki/<主題>.md`：

```markdown
# [功能／模組名稱]

> 最後更新：YYYY-MM-DD｜相關 tickets：TKT-NNN

## 現況說明
[只寫現況，不寫歷程]

## 設計決策
[為何這樣做，取捨考量]

## 已知限制
[邊界案例、尚未解決的問題]

## 實作 Know-How
[踩過的坑、重要細節、程式技巧]

## 相關檔案
- `path/to/file.py` — 說明
```

3. 更新 `docs/wiki/README.md` 索引
4. 更新 ticket 的 `related_wiki` 欄位（**票留在 `done/`，不移動**）
5. 「修改歷程」補：`- YYYY-MM-DD \`[Wiki 同步]\` 同步至 docs/wiki/XXX.md`

---

## 模式 5：列表模式（status）

掃描各子資料夾，依序輸出：

```
📋 Ticket 總覽

[todo]
  TKT-004  [feature/medium]  取消委託事件記錄至 transactions
  TKT-005  [feature/medium]  重掛單利用 snapshot polling

[in-progress]
  （空）

[in-review]
  （空）

[done]
  TKT-001  [bug/high]    EMS 啟動崩潰 — Unknown method 'auction'
  TKT-002  [task/medium] 移除 strategy_decision_ts 欄位
  TKT-003  [bug/medium]  exchange_report_ts 大量 NULL

[archive]  ← 暫時擱置，不在開發計劃中
  （空）
```

---

## 全域規則

- **所有 ticket 與 wiki 內容一律以繁體中文撰寫**
- **建票完成前不寫任何程式碼**
- **每次完成一個動作後，立即在「修改歷程」補一筆摘要**
- **票的位置即狀態**，front-matter `status` 與資料夾必須同步
- **wiki 只從 `done/` 票同步**，`archive` 票不同步 wiki
- Priority：`high`（系統崩潰/交易錯誤）、`medium`（功能缺失）、`low`（清理/體驗）
- 需求不清楚時，description 寫「**待釐清**：[問題]」，不臆測

---

## 執行前必做

1. Glob `docs/tickets/**/*.md` 確認現有票號與位置
2. 建票後 Read 驗證檔案內容正確

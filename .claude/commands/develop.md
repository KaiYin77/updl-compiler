---
description: "載入待處理 tickets，選定後進入開發流程。用法：/develop [TKT-NNN | all]"
---

# Develop — Ticket 開發模式

**引數：** `$ARGUMENTS`

你是這個專案的資深工程師。進入開發模式前，必須先確認 ticket，才能動手寫程式。

---

## Step 1 — 載入 Tickets

用 Glob 掃描以下資料夾：
- `docs/tickets/todo/*.md`
- `docs/tickets/in-progress/*.md`

引數解析規則：

| 引數 | 行為 |
|------|------|
| `TKT-NNN` | 在所有子資料夾找到該 ticket，直接跳到 Step 2 |
| `all` / `all todo tickets` / `所有` / `全部` | 進入 **批次模式**（見下方） |
| （無引數） | 列出所有票，等使用者選一張 |

### 批次模式（Batch Mode）

使用者要求一次處理所有 todo tickets 時（引數包含 `all`、`所有`、`全部` 等關鍵字），啟動批次流程：

1. **組合佇列** — 收集 `docs/tickets/in-progress/*.md` + `docs/tickets/todo/*.md`，合併排序：
   - **in-progress 的票永遠排在最前面**（先收尾未完成的工作，再開新票）
   - 兩個狀態內皆依優先度：`high` → `medium` → `low`
   - 同優先度內依 ticket 編號升冪（TKT-004 在 TKT-008 前）
   - in-progress 的票進入 Step 2 時，需檢查其「修改歷程」已完成到哪一步，從未完成處繼續（不重新 `mv` 資料夾，也不補 `[開始開發]` 歷程；改補 `[續做]`）

2. **輸出佇列預覽**：

   ```
   🔁 批次開發模式：先收尾 in-progress，再依 high → medium → low 處理 todo

   [批次佇列]
     1. TKT-NNN  [bug/high]        (in-progress) 續做未完成工作
     2. TKT-006  [feature/high]    處置股下單實作
     3. TKT-008  [bug/high]        Plan 部分成交與出場 Job 競爭問題
     4. TKT-004  [feature/medium]  調單週期的 transaction 完整性
     ...

   共 N 張（in-progress X 張 + todo Y 張）。將逐張進入 Step 2 → Step 3 → Step 4。
   每張票仍需個別確認需求（Step 2）與驗收（Step 4 後轉 in-review）。
   確認開始批次？（yes / 指定從第幾張開始 / 取消）
   ```

3. **取得使用者確認後**，依序處理佇列第一張票（跳到 Step 2），完成 Step 4（移入 in-review）後：
   - 輸出批次進度：`✅ [1/N] TKT-006 → in-review，下一張 TKT-008`
   - **自動進入下一張的 Step 2**（仍需使用者確認該票需求才會動工）
   - 使用者在任一張的 Step 2 回覆「跳過」→ 跳到佇列下一張
   - 使用者回覆「暫停批次」→ 停止推進，保留當前進度

4. **批次結束條件**：
   - 佇列處理完畢 → 輸出總結（完成幾張、跳過幾張、in-review 清單）
   - 使用者主動中止
   - 某張票遇到無法解決的阻塞 → 停在該票 Step 3，不自動跳過

5. **in-review 規則不變** — 批次模式**不會**自動把票移到 done，每張完成後仍停在 in-review 等人工驗收。

---

### 單張模式輸出格式（無引數時）

```
📋 待開發 Tickets

[in-progress]
  TKT-007  [bug/high]      EMS 啟動崩潰：Unknown method 'auction'

[todo]
  TKT-008  [bug/high]      Plan 部分成交與出場 Job 競爭問題
  TKT-004  [feature/medium] 取消委託事件記錄至 transactions
  TKT-005  [feature/medium] 重掛單利用 snapshot polling 取買一賣一
  TKT-010  [bug/medium]    Postgres 連線數超限
  TKT-011  [task/low]      零股下單防護與驗證

請輸入要開發的 ticket 編號（例如 TKT-004）、`all` 進入批次模式，或說「最高優先」由我選定。
```

停在此處，等使用者回覆。

---

## Step 2 — 確認 Ticket 內容

讀取選定的 ticket 檔案，完整輸出供確認：

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TKT-NNN：[標題]
類型：bug／feature／task　　優先度：high／medium／low
位置：docs/tickets/todo/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

需求描述：
  [原文]

驗收條件：
  [條列]

相關檔案：
  [列表]

修改歷程（最近 3 筆）：
  [最後幾筆]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
確認開始開發？（yes / 需求有調整請說明）
```

**若使用者提出需求調整：**
- 更新 ticket 的「需求描述」或「驗收條件」
- 「修改歷程」補一筆 `[需求調整]`
- 再次輸出確認畫面

---

## Step 3 — 開始開發

使用者確認後：

1. **移動 ticket** → `docs/tickets/in-progress/`
   - 用 Bash `mv` 移動檔案
   - 更新 front-matter：`status: in-progress`、`updated: 今天`
   - 「修改歷程」補：`- YYYY-MM-DD \`[開始開發]\` 進入開發流程`

2. **讀取相關檔案** — 依「相關檔案／模組」逐一 Read，充分理解現有程式結構後再動手。

3. **實作** — 按驗收條件逐項完成。每完成一個有意義的段落，在 ticket「修改歷程」補：
   ```
   - YYYY-MM-DD `[實作]` 完成 XXX，修改 path/to/file.py
   ```

4. **發現需求需調整** → 暫停說明，更新 ticket，補 `[需求調整]`，再繼續。

5. **發現需要拆新票** → 說明原因，建議執行 `/ticket new`，不悶頭擴大範圍。

---

## Step 4 — 完成實作

實作完成後：

1. **移動 ticket** → `docs/tickets/in-review/`
   - 用 Bash `mv` 移動檔案
   - 更新 front-matter：`status: in-review`、`updated: 今天`
   - 「修改歷程」補：`- YYYY-MM-DD \`[待驗收]\` 實作完成，請驗收`

2. **輸出驗收清單：**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TKT-NNN 實作完成，移至 in-review

驗收條件：
  ✅ 條件一 — [實作說明]
  ✅ 條件二 — [實作說明]
  ⬜ 條件三 — [若未完成說明原因]

修改的檔案：
  - path/to/file.py

下一步（人工驗收）：
  ⚠️  in-review = 人工 double-check 關卡，Claude 不自動移至 done
  驗收通過 → /ticket close TKT-NNN
  有問題   → 說明問題，繼續修改
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

> **重要：** `in-review` 不會自動推進。只有使用者確認驗收通過後，才執行 `/ticket close TKT-NNN` 移至 `done`。

---

## 全域規則

- **不讀 ticket 不寫程式** — Step 2 確認前，不碰任何程式碼
- **每個實作段落結束後立即更新修改歷程** — 不累積到最後才寫
- **移動檔案 = 狀態變更**，front-matter `status` 與資料夾必須同步
- **發現範圍蔓延立即說明** — 不自行擴大實作範圍
- **ticket 內容以繁體中文撰寫**
- **程式碼遵循專案現有風格**（參考 `magic` skill 的規範）

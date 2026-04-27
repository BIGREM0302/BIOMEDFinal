## GITHUB 指令

---

# 🚀 【三人 Git 協作標準流程】

## 🧱 一、開發前（每個人第一次設定）

### 1️⃣ clone 專案

```bash
git clone <repo-url>
cd <repo-name>
```

---

### 2️⃣ 初始化 uv 環境

```bash
uv venv
uv sync
```

👉 這會：

- 建立 `.venv`
- 安裝 `uv.lock` 裡的所有套件

---

### 3️⃣ 建立自己的 branch（非常重要❗）

每個人用自己的名字，例如：

```bash
git checkout -b feature/max
git checkout -b feature/alice
git checkout -b feature/bob
```

👉 命名建議：

```
feature/<名字或功能>
bugfix/<功能>
```

---

### 4️⃣ 推上 GitHub

```bash
git push -u origin feature/max
```

---

# 🔧 二、開發中（每天都會用）

## ⭐ 每次開始寫 code 前

👉 **先同步 main（避免爆炸）**

```bash
git checkout main
git pull origin main
```

再切回你的 branch：

```bash
git checkout feature/max
git merge main
```

---

## 💻 寫 code + commit

```bash
git add .
git commit -m "feat: add login API"
```

---

## 📤 push 到 GitHub

```bash
git push
```

---

## 📦 有人新增套件時（很重要）

如果 repo 有變：

```bash
git pull
uv sync
```

👉 因為：

- `pyproject.toml` 或 `uv.lock` 可能更新

---

# 🔀 三、開發後（準備合併）

## 1️⃣ 再同步一次 main（避免衝突）

```bash
git checkout main
git pull origin main

git checkout feature/max
git merge main
```

---

## 2️⃣ 解 conflict（如果有）

👉 修改衝突檔案後：

```bash
git add .
git commit
```

---

## 3️⃣ push branch

```bash
git push
```

---

## 4️⃣ 開 Pull Request（PR）

到 GitHub：

👉 `feature/max → main`

---

## 5️⃣ 合併（通常由一人負責）

Merge PR → main

---

# 🔄 四、合併後大家要做的事

每個人都要：

```bash
git checkout main
git pull origin main
```

然後：

```bash
uv sync
```

---

# 🧹 五、清理 branch（可選）

```bash
git branch -d feature/max
git push origin --delete feature/max
```

---

# 📁 六、.gitignore 建議（uv 專案）

```gitignore
.venv/
__pycache__/
*.pyc
.env
```

---

# 📌 七、一定要 commit 的檔案

```text
✅ pyproject.toml
✅ uv.lock   ← 超重要（鎖版本）
❌ .venv/
```

---

# ⚠️ 八、常見爆炸點（直接幫你們避雷）

### ❌ 1. 直接在 main 開發

👉 一定要 branch！

---

### ❌ 2. 忘記 pull 就寫

👉 會 conflict 地獄

---

### ❌ 3. 沒 uv sync

👉 套件錯版本 → 跑不起來

---

### ❌ 4. force push main

👉 團隊直接爆掉 ☠️

---

# 🧠 九、簡化版流程（給懶人）

每天：

```bash
git checkout main
git pull

git checkout feature/你的名字
git merge main

# 開發
git add .
git commit -m "xxx"
git push
```

---

# 🧩 十、如果你們想更專業（可選）

可以加：

- pre-commit（自動 lint）
- CI/CD（GitHub Actions）
- commit message 規範（feat/fix）

---

# 🧾 一句話總結

👉 **每個人用自己的 branch → 常 pull main → 用 PR 合併 → uv sync 保持環境一致**

---

## 1. 專案概述

本專案提供一個基於 Python 的基準機器學習程式，用於分析 EEG 訊號，並將其分類為 **「放鬆 (Relax)」、「專注 (Focus)」與「眨眼 (Blink)」** 三種狀態。

本學期使用 **BrainLink 腦波儀 (單通道 Fp1, 取樣率 512 Hz)**。每組需使用自行錄製的腦波資料來訓練模型。我們將使用 **MLP** 模型，並透過 **Leave-One-Subject-Out (LOSO)** 的方式來評估效能。

👉 任務：

- **不能更換模型類型**（必須維持使用 `MLPClassifier`）
- 可以 **調整 MLP 的超參數** (隱藏層結構、學習率、正則化等)
- 設計並實作合適的 **資料前處理 (Preprocessing) 與後處理 (Postprocessing)** 以提升分類準確率。
- **實驗目標：Overall Mean Accuracy ≥ 65%**。

## 2. 環境設定 (使用 `uv`)

本專案範例使用 Python 套件管理器 **`uv`** 來建立虛擬環境與安裝套件。

1. **安裝 `uv`** (若尚未安裝)：
   - Windows (PowerShell): `irm https://astral.sh/uv/install.ps1 | iex`
   - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. **建立虛擬環境**：
   開啟終端機並切換到專案資料夾，輸入：
   ```bash
   uv venv
   uv sync
   ```
   即可複製同環境並執行 `main.py`。

也可以使用其他方法建置環境，確保檔案能執行即可。

## 3. 檔案結構

請確保專案資料夾結構如下，程式與資料需放在同一目錄下。
EEG 資料命名格式說明：`組員學號_類別_回合.txt`。

```bash
your_project_folder/
├── brainlink.exe          # （提供）蒐集資料的執行檔，不用連同作業繳交
├── main.py                # （提供）主要 Python 程式
├── pyproject.toml         # （提供）供 uv 複製環境
├── uv.lock                # （提供）供 uv 複製環境
└── bci_dataset_114-2/     # 新增你這組當受試者資料的資料夾
    ├── 學號1(小寫)/
    │   ├── 學號1_1_1.txt   # 放鬆狀態第 1 回合
    │   ├── 學號1_1_2.txt
    │   ├── ... (共30回合)
    │   ├── 學號1_2_1.txt   # 專注狀態第 1 回合
    │   └── ... (共30回合)
    │   ├── 學號1_3_1.txt   # 眨眼狀態第 1 回合
    │   ├── ... (共30回合)
    │   └── ... (共90個檔案)
    ├── 學號2(小寫)/
    │   └── ... (共90個檔案)
    └── 學號3(小寫)/
        └── ... (共90個檔案)
```

- 每個 `.txt` 檔為單欄 EEG 時域數值訊號，代表一回合 **20秒** 的資料。
- 取樣率：512 Hz (BrainLink 預設，每檔案應有約 10240 筆數據)。
- `_1_` 代表 放鬆狀態 (Relax)，類別標籤為 0。
- `_2_` 代表 專注狀態 (Focus)，類別標籤為 1。
- `_3_` 代表 眨眼狀態 (Blink)，類別標籤為 2。

## 4. 可修改與不可修改的 HPs

| 類別           | 參數                 | 是否可調整 | 建議範圍與限制                           |
| -------------- | -------------------- | ---------- | ---------------------------------------- |
| **模型結構**   | `HIDDEN_LAYER_SIZES` | ✅         | 可自由設定，如 (128, 64, 32)、(256, 128) |
|                | `activation`         | ❌         | 固定為 `relu`                            |
|                | `solver`             | ❌         | 固定為 `adam`                            |
| **訓練超參數** | `LEARNING_RATE_INIT` | ✅         | 0.005 ~ 0.02                             |
|                | `ALPHA` (L2 正則化)  | ✅         | 0.0001 ~ 0.05                            |
|                | `BATCH_SIZE`         | ✅         | 32 ~ 128                                 |
|                | `MAX_ITER`           | ✅         | 50 ~ 200                                 |
| **資料切片**   | `SAMPLING_RATE`      | ❌         | 固定為 512                               |
|                | `SEGMENT_LENGTH`     | ✅         | 2 ~ 6 秒                                 |
|                | `OVERLAP_RATIO`      | ✅         | 0.0 ~ 0.8                                |
| **其他**       | `early_stopping`     | ❌         | 固定不開放                               |

## 5. 程式修改提示

請在程式碼中搜尋 `# === STUDENT PREPROCESSING HERE ===` 的區塊。
原始 EEG 時域訊號包含極大的眨眼雜訊與肌電干擾，建議實作以下方法：

1. **前處理**：例如濾波 (1–40 Hz)、z-score 標準化、band power 特徵。
2. **後處理**：例如多數決投票、使用 `predict_proba` 調整閾值。

⚠️ 請在程式碼中標明修改區域，例如：

```python
# === student preprocessing ===
# === student postprocessing ===
```

## 6. 輸出結果解讀

### a. 終端機輸出 (範例)

```bash
...
==================================================
Overall Mean Accuracy: 0.552 ± 0.091

[Relax Class]:
  - Accuracy (Recall): 0.561 (2234/3983)
  - Precision: 0.544 (2234/4100)

[Focus Class]:
  - Accuracy (Recall): 0.543 (2201/4050)
  - Precision: 0.561 (2201/3920)

[Blink Class]:
  - Accuracy (Recall): 0.543 (2201/4050)
  - Precision: 0.561 (2201/3920)

Results saved to 'bci_results_raw_data.png'
```

### b. 圖片輸出

1. 每位受試者的準確率長條圖
2. 整體混淆矩陣 (3x3)
3. 訓練損失曲線

## 7. 評分標準

- **規範遵守 (20%)**：模型未被非法修改
- **準確率 (40%)**：整體準確率達到 ≥65% 拿 30%，≥70% 拿 40%
- **報告分析 (40%)**：詳細說明方法與比較結果

## 8. 繳交報告

```bash
114_2_gX_exp3.zip/
├── 114_2_gX_exp3.pdf       # 報告
├── main.py                 # 修改完的程式
└── bci_dataset_114-2/      # 放你這組當受試者資料的資料夾
    ├── 學號1(小寫)/
    │   ├── 學號1_1_1.txt   # 放鬆狀態 (Relax) 第 1 回合
    │   ├── 學號1_1_2.txt
    │   ├── ... (共30回合)
    │   ├── 學號1_2_1.txt   # 專注狀態 (Focus) 第 1 回合
    │   └── ... (共30回合)
    │   ├── 學號1_3_1.txt   # 眨眼狀態 (Blink) 第 1 回合
    │   ├── ... (共30回合)
    │   └── ... (共90個檔案)
    ├── 學號2(小寫)/
    │   └── ... (共90個檔案)
    └── 學號3(小寫)/
        └── ... (共90個檔案)
```

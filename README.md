# Connect 4 AI (AlphaZero-style)

這是一個基於蒙地卡羅樹搜尋 (MCTS) 與卷積神經網路 (CNN) 的四子棋 (Connect 4) AI 專案。本專案包含完整的強化學習訓練管線：從自我對弈生成棋譜、CNN 模型訓練，到結合 Minimax 搜尋的實戰推論引擎，並特別支援打包成「無套件依賴的純 NumPy 版本」以供提交至 Kaggle ConnectX 競賽。

---
## ?? 專案目錄結構 (Project Structure)

以下為本專案在生醫實驗資料夾下的精簡結構，省略了部分快取與生成的數據檔案：

```text
? BiomedFINAL/ 
├── ? README.md
├── ? exp3/                       # 上次實驗檔案，以一資料夾表示
└── ? Connect-4-AI-main/          # 本次期末 AI 專案核心
    │
    ├── ? monted/                 # 訓練完成的 TensorFlow 實體模型
    │
    ├── ? CNNPlayWithSearch.py    # 本地對戰與推論主程式
    ├── ? Connect4CnnModel.py     # CNN 訓練腳本
    ├── ? MonteGameGen.py         # 蒙地卡羅自我對弈與數據生成腳本
    ├── ? connect4.py             # 四子棋環境與 Pygame 介面
    ├── ? newMonte.py             # MCTS 演算法核心
    │
    ├── ? extract_weights.py      # NumPy 權重萃取工具
    │
    └── ? Kaggle_submission/      # Kaggle 提交檔案
        ├── ? main.py             # Kaggle 提交專用入口腳本
        │
        ├── ? treelib/            # 供 Kaggle 使用的獨立依賴套件（複製自.venv）
        └── ? weights.npz         # 萃取出的純 NumPy 權重 (生成)
```

---

## ?? 重要前提：資料夾路徑

本專案與其他實驗（如 exp3）的檔案放置在同一個大目錄下。在執行任何指令前，**請務必先在終端機 (Terminal) 中進入本專案的專屬資料夾**，以避免讀取到錯誤的環境或檔案：

```bash
cd Connect-4-AI-main
```

---

## ?? 環境建置與依賴安裝

如果你已經擁有訓練好的模型（確認目錄下有一個名為 `monted` 的資料夾），你可以直接啟動對戰介面來測試 AI 的強度。

```bash
uv run python CNNPlayWithSearch.py
```

- *操作方式*：根據終端機的提示，輸入 `0` 到 `6` 的數字來選擇你要落子的欄位。

- *難度與速度調整*：如果覺得 AI 思考太久，可打開 `CNNPlayWithSearch.py`，找到 `treeGen` 函數，將 `counterL` 的條件（原本預設可能是 `14000`）往下調低（例如改為 `2000` 或 `4000`）。數值越高 AI 越強，但運算時間呈指數成長。

---

## ? 第二階段：如何重新訓練 AI 大腦

如果你想讓 AI 變得更聰明，或者你剛下載專案還沒有 `monted` 資料夾，請按照以下順序執行：

### 步驟 2a：生成對弈數據 (產生教材)
執行 `MonteGameGen.py`，程式會利用純蒙地卡羅樹搜尋讓電腦瘋狂自我對弈，並記錄下盤面狀態與勝負結果。

```Bash
uv run python MonteGameGen.py
```

#### ? 訓練注意事項：

程式預設使用多處理序平行運算。開始前，請確保程式碼最下方的生成區塊 (`f(x)` 與 `Pool`) 是未被註解的狀態。

讓它跑一段時間，直到產出約 200MB 以上的 `.csv` 檔（包含 `0.csv`、`1.csv`、`neg1.csv`）。

資料合併：數據生成完畢後，手動刪除這三個 CSV 檔的「最後一行空白行」（避免 Bug），然後註解掉生成程式碼，並取消註解檔案最底下的 Pandas 合併邏輯，再次執行該檔案，產出最終教材 `final.csv`。

### 步驟 2b：訓練 CNN 模型 (學生學習)
有了 `final.csv`，就可以開始訓練卷積神經網路。

```Bash
uv run python Connect4CnnModel.py
```

程式會將 CSV 裡的棋盤轉換為圖像張量，並餵給神經網路。

訓練完成後，專案目錄下會自動生成一個名為 `monted` 的資料夾，這就是你的新 AI 大腦。

---

## ? 第三階段：提交至 Kaggle (純 NumPy 脫殼版)

由於 Kaggle 官方的 Python 評估環境存在 TensorFlow C++ 與 NumPy 2.x 的底層衝突，直接上傳 TensorFlow 程式碼會導致伺服器崩潰。因此，我們必須將模型轉換為「純 NumPy 陣列」。

### 步驟 3a：萃取模型權重
在本地端執行權重萃取腳本，把 `monted` 資料夾內的神經網路參數抽出來，存成輕量的壓縮檔。

```Bash
uv run python extract_weights.py
```
(成功後，目錄下會多出一個 weights.npz 檔案)

### 步驟 3b：準備提交檔案
建立一個全新的、乾淨的暫存資料夾（例如 kaggle_submission），並只放入以下三個東西：

- `main.py`：(已整合純 NumPy 卷積運算與 Kaggle agent 介面的最終版腳本)

- `weights.npz`：(剛剛萃取出來的純數值大腦)

- `treelib/`：(從虛擬環境 .venv/Lib/site-packages/ 裡複製過來的整個依賴資料夾)

### 步驟 3c：打包壓縮

打開終端機，`cd` 進入你剛剛建立的 `kaggle_submission` 資料夾，執行以下壓縮指令（Windows 10/11 內建支援）：

```Bash
tar -czvf submission.tar.gz main.py weights.npz treelib
```

最後，將產出的 `submission.tar.gz` 上傳到 Kaggle，大功告成！

## ? 參考資料 (References)

本專案的演算法架構與部署策略，參考了以下文獻與開源工具：

* **Kaggle 競賽平台**
  * [Kaggle ConnectX 官方環境](https://www.kaggle.com/c/connectx): 四子棋環境設計與 Agent 提交規範。

* **Github**
  * [Github](https://github.com/CanProjects/Connect-4-AI): 原始模型與 code 來源
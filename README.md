# 🧠 Brain-Computer Interface (BCI) Connect-4 Game

> **An advanced BCI system integrating deep learning, game AI, and hardware control for real-time brain signal classification and interactive gameplay.**

---

## 📋 Overview

This is a comprehensive BCI project combining:

- **Deep Learning Models**: Multiple CNN architectures for EEG signal classification
- **Game AI**: Connect-4 game with Monte Carlo Tree Search (MCTS) and CNN-based board evaluation
- **Hardware Integration**: Arduino-based hardware control with real-time signal processing
- **BCI Simulation**: MockBCI for testing and development without hardware

The project supports both **training-only** modes (for model development) and **interactive game modes** (with real or simulated BCI).

---

## 📁 Project Structure

```
BioMed_Final/
├── 🤖 Training & Core Models (Root Level)
├── 🎮 game/              # Connect-4 Game + BCI Integration
├── ⚙️  hw/               # Arduino Hardware Debug Files
├── 📊 bci_dataset_*/     # EEG Datasets
└── 📂 runs/              # Training Results & Logs
```

---

## 🤖 Training & Model Files

### Core Training Scripts

| File                 | Description                              | Key Features                                             |
| -------------------- | ---------------------------------------- | -------------------------------------------------------- |
| **main.py**          | Basic 1D-CNN + Squeeze-Excitation Module | Fast baseline, ~512 input samples                        |
| **deep_main.py**     | CNN + Handcrafted Features               | Feature fusion with signal statistics (kurtosis, skew)   |
| **dual_main.py**     | Dual-Model Ensemble                      | Combines CNN and feature-based predictions               |
| **focal_main.py**    | Focal Loss-based Training                | Better handling of class imbalance                       |
| **xgboost_train.py** | XGBoost Baseline                         | Gradient boosting with hyperparameter optimization (HPO) |
| **modify_main.py**   | Custom Modified Version                  | Experimental variations                                  |

### Utility Scripts

| File            | Purpose                                    |
| --------------- | ------------------------------------------ |
| **plot_eeg.py** | Visualize raw EEG signals and spectrograms |

### Key Features (All Models)

- ✅ **LOSO Cross-Validation**: Leave-One-Subject-Out for generalization testing
- ✅ **Artifact Rejection**: Automatic removal of noisy/corrupted signals (threshold: 600 µV)
- ✅ **Class Weighting**: Balanced training on imbalanced datasets
- ✅ **Early Stopping**: Prevent overfitting with patience-based stopping
- ✅ **Subject-Specific Calibration**: Personalized scaling per subject

### Configuration Parameters

All models use `Config` class with:

```python
DATASET_PATH = "bci_dataset_114-2_any"
SAMPLING_RATE = 512  # Hz
WINDOW_SECONDS = 1.0-2.0
STEP_SECONDS = 0.25-1.0
ARTIFACT_THRESHOLD = 600  # µV
BATCH_SIZE = 128
EPOCHS = 60
DEVICE = cuda/cpu (auto-detected)
```

### Output Structure

Training runs create timestamped directories:

```
runs/YYYYMMDD_HHMMSS/
├── best_model.pth          # Best checkpoint
├── metrics.json            # Accuracy, Precision, Recall, F1
├── confusion_matrix.png    # Visual evaluation
└── training_log.txt        # Detailed run info
```

---

## 🎮 Game Folder

### Main Game

| File                         | Purpose                               | Integration                 |
| ---------------------------- | ------------------------------------- | --------------------------- |
| **connect4.py**              | Main game engine                      | Pygame UI + Gym environment |
| **connect4_hw.py**           | Hardware-connected version            | Real hardware I/O           |
| **CNNPlayWithSearch.py**     | AI player combining CNN + tree search | Smart opponent              |
| **CNNPlayWithSearch_old.py** | Legacy CNN search algorithm           | Reference version           |

### AI & Game Logic

| File                    | Purpose                                                    |
| ----------------------- | ---------------------------------------------------------- |
| **Connect4CnnModel.py** | CNN model for board position evaluation                    |
| **MonteGameGen.py**     | Monte Carlo Tree Search (MCTS) + self-play data generation |
| **newMonte.py**         | Improved Monte Carlo implementation                        |

### BCI Integration

| File               | Purpose                                     | Status                 |
| ------------------ | ------------------------------------------- | ---------------------- |
| **mock_bci.py**    | Simulates brain signals (focus/relax/blink) | ✅ Development/Testing |
| **real_bci.py**    | Real hardware BCI signal processing         | 🔌 Hardware-dependent  |
| **test_serial.py** | Debug serial communication                  | Testing tool           |

### Assets & Models

| File                                     | Purpose                                             |
| ---------------------------------------- | --------------------------------------------------- |
| **bci_model_final.pth**                  | Pre-trained BCI CNN classifier                      |
| **feature_scaler.pkl**                   | Scikit-learn StandardScaler (feature normalization) |
| **ai_avatar.png**, **player_avatar.png** | Cyberpunk UI graphics                               |
| **model/**                               | Additional models directory                         |
| **monted/**                              | MCTS game tree data                                 |
| **old/**                                 | Legacy code versions                                |

### Requirements

```bash
cd game
cat requirements.txt
```

Key dependencies:

- `pygame` - Game rendering
- `torch` - CNN inference
- `gym` - RL environment
- `stable-baselines3` - AI training utilities
- `scipy`, `numpy` - Signal processing
- `xgboost` - Alternative classifier

### Running the Game

```bash
# Start with simulation
python connect4.py

# Hardware version (requires Arduino connection)
python connect4_hw.py

# Mock BCI testing
python mock_bci.py
```

---

## ⚙️ Hardware Folder (Arduino)

Arduino debugging and control scripts for physical game board integration.

### Debug Modules

| File                | Target Hardware       | Purpose                  |
| ------------------- | --------------------- | ------------------------ |
| **array_debug.ino** | Microcontroller       | Test array operations    |
| **board_debug.ino** | Game board controller | Main board debugging     |
| **led_debug.ino**   | LED status display    | Visual feedback testing  |
| **motor_debug.ino** | Motor/stepper control | Motion control debugging |

### Additional Directories

```
lcd_debug/      - Liquid Crystal Display debugging
board_render/   - Board rendering logic
```

### Purpose

These modules support:

- Real-time board display and LED feedback
- Motor control for physical game pieces
- Hardware state verification
- Serial communication with Python backend

---

## 📊 Data & Results

### Datasets

- **bci_dataset_114-2_any/**: Main EEG dataset (cleaned, normalized)
- **bci_dataset_114-2_any_old/**: Previous version (reference)
- **new_g3_bci_dataset/**: Alternative dataset format

### Subjects

- **b12901016/, b12901035/, ...**: Individual subject data folders
  - Files named as: `<subject>_<session>_<trial>.txt`
  - Format: Raw EEG samples (multi-channel, 512 Hz sampling rate)

### Training Results

- **runs/**: LOSO validation results, confusion matrices, metrics
- **runs_3/**: Deep model training outputs
- **bci_model_final.pth**: Final pre-trained model checkpoint

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone & enter repo
git clone <repo-url>
cd BioMed_Final

# Initialize UV environment
uv venv
uv sync
```

### 2. Train a Model

```bash
# Basic CNN model
python main.py

# CNN + Handcrafted features
python deep_main.py

# XGBoost baseline
python xgboost_train.py

# View results
# Check runs/YYYYMMDD_HHMMSS/ for metrics and plots
```

### 3. Play the Game (Simulation)

```bash
cd game
python mock_bci.py      # Test MockBCI
python connect4.py      # Play game
```

### 4. Visualize EEG Data

```bash
python plot_eeg.py
```

---

## 📈 Model Architecture Overview

### Deep Learning Models

**1D-CNN with Squeeze-Excitation (SE)**

```
Input (1, 512)
  ↓
Conv1D blocks (16→32→64 filters)
  ↓
Squeeze-Excitation layers (channel attention)
  ↓
Global Average Pooling
  ↓
Dense (128 → output_classes)
```

**CNN + Handcrafted Features**

```
Raw Signal
  ├─→ CNN branch (1D convolutions)
  │     ↓
  │   Global features (context)
  │
  └─→ Feature Engineering branch
        - Mean, Std, Kurtosis, Skewness
        - Frequency domain (PSD, spectral entropy)
        ↓
Concatenate & fuse
  ↓
Dense classifier
```

**XGBoost**

- Feature-only input (no raw signals)
- Gradient boosting with 100+ rounds
- Hyperparameter optimization via RandomizedSearch
- Subject-specific calibration

---

## 📊 Performance Metrics

Models are evaluated using:

- **Accuracy**: Overall correctness
- **Precision/Recall**: Per-class performance
- **F1-Score**: Harmonic mean
- **Confusion Matrix**: Visual error analysis
- **LOSO Validation**: Subject-independent generalization

Expected performance (on clean data):

- CNN models: 85-95% accuracy
- XGBoost: 80-90% accuracy
- Depends heavily on data quality and subject

---

## 🔧 Development & Collaboration

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/<your-name>

# Before pushing, sync with main
git checkout main
git pull origin main
git checkout feature/<your-name>
git merge main

# Push & create PR
git push -u origin feature/<your-name>
```

### Key Commands

```bash
# Check environment
uv sync

# Run training with GPU monitoring
watch -n 1 nvidia-smi

# Clean old results
rm -rf runs/old_*/
```

---

## 📝 Citation & References

- **EEG Signal Processing**: Scipy signal module
- **Deep Learning**: PyTorch with Squeeze-Excitation networks
- **Game AI**: MCTS + Minimax algorithms
- **Baseline**: XGBoost gradient boosting

---

## 👥 Team Notes

- **Primary Dataset**: bci_dataset_114-2_any (114 subjects × 2 sessions × ~30 trials)
- **Excluded Subjects**: S06, S14, S09, S05 (poor data quality)
- **Hardware**: Arduino-based with LED/Motor feedback
- **Production Model**: bci_model_final.pth (pre-trained, ready for deployment)

---

## 📞 Support

For issues or questions:

1. Check training logs in `runs/*/training_log.txt`
2. Review confusion matrices for error patterns
3. Verify data integrity in dataset folders
4. Test with MockBCI before hardware integration

---

**Last Updated**: June 2026  
**Project Status**: ✅ Active Development
git pull
uv sync

````

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
````

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

# 🧠 Brain-Computer Interface (BCI) Connect-4 Game

> **An advanced BCI system integrating deep learning, game AI, and hardware control for real-time brain signal classification and interactive gameplay.**

---

🎥 **Hardware & Software Live Demo Video:** [https://youtu.be/mero_8mWpXw](https://youtu.be/mero_8mWpXw)

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
- **Game AI**: MCTS + Minimax algorithms, based on [CanProjects/Connect-4-AI](https://github.com/CanProjects/Connect-4-AI)
- **Baseline**: XGBoost gradient boosting
  
---

## 👥 Team Notes

- **Primary Dataset**: bci_dataset_114-2_any (18 subjects × 3 tasks × 10 rounds)
- **Excluded Subjects**: S06, S09, S14 (poor data quality)
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

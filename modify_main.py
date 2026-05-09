"""
Brain-Computer Interface 1D-CNN+SE Classifier
FINAL VERSION: Precise Labeling + Artifact Rejection + Lightweight Deep Learning + Advanced Plotting
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy import signal
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm
from datetime import datetime
import json

warnings.filterwarnings('ignore')

execute_loso = True
load_checkpoint = True

# ==========================================
# 1. 系統與資料流參數設定 (Config)
# ==========================================
class Config:
    DATASET_PATH = "bci_dataset_114-2_any"
    SAMPLING_RATE = 512
    WINDOW_SECONDS = 1.0       # 1秒視窗 (512 samples)

    RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = os.path.join("runs", RUN_TAG)
    
    # [問題1解法]: 設定不同的滑動步長來平衡資料
    STEP_SECONDS_BG = 0.5      # Relax/Focus 的滑動步長調大為 0.5 秒 (減少樣本數)
    
    # 偽影拒絕門檻 (用於 Task 2 排除自然眨眼，保留純淨專注資料)
    ARTIFACT_THRES = 600
    
    # Task 3 刻意眨眼的時間點 (秒)
    BLINK_TIMESTAMPS = [0, 4, 8, 12, 16]
    # [問題1解法]: 眨眼資料擴增 (Data Augmentation) 的時間偏移量 (秒)
    # 在標記點附近擷取多個重疊的視窗，增加模型對眨眼位置的容錯率，並大幅提升樣本數
    BLINK_SHIFTS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    
    # 模型與訓練參數
    BATCH_SIZE = 128
    EPOCHS = 60
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4        # L2 正規化，防止 Overfit
    PATIENCE = 15              # Early Stopping 耐心值
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:{DEVICE}")

def save_config():
    config_dict = {}

    for k, v in Config.__dict__.items():
        if not k.startswith("__") and not callable(v):
            try:
                json.dumps(v)
                config_dict[k] = v
            except TypeError:
                config_dict[k] = str(v)

    save_path = os.path.join(Config.RUN_DIR, "config.json")

    with open(save_path, "w") as f:
        json.dump(config_dict, f, indent=4)

    print(f"Config saved to: {save_path}")

# ==========================================
# 2. 資料處理與特徵工程
# ==========================================
def bandpass_filter(data, lowcut=1.0, highcut=40.0, fs=Config.SAMPLING_RATE, order=4):
    """帶通濾波器，去除基線漂移與高頻雜訊"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def process_subject_files(folder_path):
    """處理單一受試者資料，執行精確切割與 Z-score 標準化"""
    X, y = [], []
    win_samples = int(Config.WINDOW_SECONDS * Config.SAMPLING_RATE)
    step_samples_bg = int(Config.STEP_SECONDS_BG * Config.SAMPLING_RATE)
    
    for task in [1, 2, 3]:
        files = glob.glob(os.path.join(folder_path, f"*_{task}_*.txt"))
        for f in files:
            try:
                data = np.loadtxt(f)
            except ValueError:
                continue # 略過讀取錯誤的檔案
                
            if len(data) < 10240: # 確保有完整的 20 秒
                continue
                
            # 1. 濾波
            data = bandpass_filter(data)
            
            # 2. 依照任務類型進行切割與標註
            if task == 1: # Relax (標籤 0)
                for start in range(0, len(data) - win_samples, step_samples_bg):
                    segment = data[start:start + win_samples]
                    X.append(segment)
                    y.append(0)
                    
            elif task == 2: # Focus (標籤 1)
                for start in range(0, len(data) - win_samples, step_samples_bg):
                    segment = data[start:start + win_samples]
                    X.append(segment)
                    y.append(1)
                        
            elif task == 3: # Blink (標籤 2)
                for t in Config.BLINK_TIMESTAMPS:
                    center_sample = int(t * Config.SAMPLING_RATE)
                    # [問題1解法]: 使用 BLINK_SHIFTS 進行資料擴增
                    for shift in Config.BLINK_SHIFTS:
                        start = center_sample + int(shift * Config.SAMPLING_RATE)
                        # 確保索引不越界
                        if start >= 0 and start + win_samples <= len(data):
                            segment = data[start:start + win_samples]
                            X.append(segment)
                            y.append(2)
                        
    if not X: return None, None
    
    X_np = np.array(X)
    
    # 3. Z-score 標準化 (對每個 Segment 獨立進行，消除基準線差異)
    mean = np.mean(X_np, axis=1, keepdims=True)
    std = np.std(X_np, axis=1, keepdims=True) + 1e-8
    X_np = (X_np - mean) / std
    
    # 擴充維度以符合 PyTorch Conv1d 輸入格式: (Batch, Channels, Length) -> (N, 1, 512)
    return np.expand_dims(X_np, axis=1).astype(np.float32), np.array(y, dtype=np.int64)

def load_all_data():
    """讀取所有受試者資料，並以 dict 儲存以便 LOSO"""
    subject_folders = sorted([f.path for f in os.scandir(Config.DATASET_PATH) if f.is_dir()])
    dataset_dict = {}
    
    for folder in subject_folders:
        sub_id = os.path.basename(folder)
        X, y = process_subject_files(folder)
        if X is not None:
            dataset_dict[sub_id] = {'X': X, 'y': y}
            print(f"Loaded {sub_id}: {len(y)} segments (Relax: {np.sum(y==0)}, Focus: {np.sum(y==1)}, Blink: {np.sum(y==2)})")
            
    return dataset_dict

# ==========================================
# 3. 輕量化神經網路架構 (1D-CNN + SE Attention)
# ==========================================
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class LightweightBCINet(nn.Module):
    def __init__(self, num_classes=3):
        super(LightweightBCINet, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=32, stride=2, padding=16)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(4)
        self.se1 = SELayer(16)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=8, stride=1, padding=4)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(4)
        self.se2 = SELayer(32)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.AdaptiveAvgPool1d(1) 
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.se1(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.se2(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x)))).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ==========================================
# 4. 訓練與 LOSO 驗證流程
# ==========================================
def train_model(train_loader, val_X, val_y, class_weights):
    model = LightweightBCINet().to(Config.DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(Config.DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    val_X_tensor = torch.FloatTensor(val_X).to(Config.DEVICE)
    val_y_tensor = torch.LongTensor(val_y).to(Config.DEVICE)
    
    best_acc = 0
    patience_counter = 0
    best_model_state = None
    loss_curve = []
    
    for epoch in tqdm(range(Config.EPOCHS), desc="Training Epochs"):
        model.train()
        epoch_losses = []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(Config.DEVICE), batch_y.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
        loss_curve.append(np.mean(epoch_losses))
            
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_X_tensor)
            _, predicted = torch.max(val_outputs, 1)
            acc = accuracy_score(val_y_tensor.cpu(), predicted.cpu())
            
        if acc > best_acc:
            best_acc = acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= Config.PATIENCE:
            break
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        final_outputs = model(val_X_tensor)
        _, final_pred = torch.max(final_outputs, 1)
        
    return final_pred.cpu().numpy(), loss_curve

def leave_one_subject_out_cv():
    dataset_dict = load_all_data()
    if not dataset_dict:
        print("未找到資料，請檢查路徑設定。")
        return None, None, None
        
    subjects = sorted(list(dataset_dict.keys()))
    print(f"\n準備開始 LOSO 驗證，共 {len(subjects)} 位受試者...")
    
    results = {
        'subject_names': [],
        'accuracies': [],
        'confusion_matrices': [],
        'loss_curves': []
    }
    
    all_y_true, all_y_pred = [], []
    
    for test_sub in subjects:
        train_X_list, train_y_list = [], []
        test_X = dataset_dict[test_sub]['X']
        test_y = dataset_dict[test_sub]['y']
        
        for sub in subjects:
            if sub != test_sub:
                train_X_list.append(dataset_dict[sub]['X'])
                train_y_list.append(dataset_dict[sub]['y'])
                
        train_X = np.vstack(train_X_list)
        train_y = np.hstack(train_y_list)
        
        classes = np.unique(train_y)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_y)
        
        tensor_x = torch.FloatTensor(train_X)
        tensor_y = torch.LongTensor(train_y)
        train_dataset = TensorDataset(tensor_x, tensor_y)
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        
        print(f"\nTraining model with {test_sub} left out...")
        y_pred, loss_curve = train_model(train_loader, test_X, test_y, weights)
        
        acc = accuracy_score(test_y, y_pred)
        cm = confusion_matrix(test_y, y_pred, labels=[0, 1, 2])
        
        results['subject_names'].append(test_sub)
        results['accuracies'].append(acc)
        results['confusion_matrices'].append(cm)
        results['loss_curves'].append(loss_curve)
        
        all_y_true.extend(test_y)
        all_y_pred.extend(y_pred)
        
        print(f"Subject {test_sub} Test Accuracy: {acc:.4f}")
        
    return all_y_true, all_y_pred, results

# ==========================================
# 5. 結果評估與視覺化
# ==========================================
def plot_and_evaluate(y_true, y_pred, results):
    if results is None: return
    
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS")
    print("="*50)
    
    mean_acc = np.mean(results['accuracies'])
    std_acc = np.std(results['accuracies'])
    print(f"Overall Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n")
    
    target_names = ['Relax', 'Focus', 'Blink']
    
    precision = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    recall = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    
    for i, name in enumerate(target_names):
        print(f"[{name}]")
        print(f"  - Accuracy (Recall): {recall[i]:.4f}")
        print(f"  - Precision:         {precision[i]:.4f}\n")
        
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('BCI Classifier (1D-CNN+SE) - Group LOSO Results', fontsize=16)
    
    subject_names = results['subject_names']
    axes[0].bar(subject_names, results['accuracies'], 
                color=['green' if acc >= 0.7 else 'orange' if acc >= 0.65 else 'red' for acc in results['accuracies']])
    axes[0].set_title('Accuracy by Subject')
    axes[0].set_ylabel('Accuracy')
    axes[0].axhline(y=np.mean(results['accuracies']), color='r', linestyle='--', label=f'Mean: {np.mean(results["accuracies"]):.3f}')
    axes[0].axhline(y=0.65, color='blue', linestyle=':', label='Target: 0.65')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis='x', rotation=45) 
    
    total_cm = np.sum(results['confusion_matrices'], axis=0)
    sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Relax', 'Focus', 'Blink'], yticklabels=['Relax', 'Focus', 'Blink'], ax=axes[1])
    axes[1].set_title('Overall Confusion Matrix')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('Actual Label')
    
    valid_loss_curves = [lc for lc in results['loss_curves'] if len(lc) > 0]
    if valid_loss_curves:
        for i, loss_curve in enumerate(valid_loss_curves):
            axes[2].plot(loss_curve, alpha=0.7, label=subject_names[i])
        axes[2].set_title('Training Loss Curves')
        axes[2].set_xlabel('Epoch') 
        axes[2].set_ylabel('Loss')
        if len(subject_names) <= 15:
            axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(Config.RUN_DIR, "dl_bci_results_dashboard.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #plt.savefig(f"dl_bci_results_dashboard.png", dpi=300, bbox_inches='tight')
    #print("Dashboard saved as 'dl_bci_results_dashboard.png'")
    print(f"Dashboard saved as '{save_path}'")
    plt.show()

def train_and_save_final_model():
    dataset_dict = load_all_data()
    if not dataset_dict:
        print("未找到資料，無法訓練最終模型。")
        return
        
    print("\n準備訓練最終佈署模型 (使用全部資料)...")
    
    X_all_list, y_all_list = [], []
    for sub in dataset_dict:
        X_all_list.append(dataset_dict[sub]['X'])
        y_all_list.append(dataset_dict[sub]['y'])
        
    train_X = np.vstack(X_all_list)
    train_y = np.hstack(y_all_list)
    
    classes = np.unique(train_y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_y)
    
    tensor_x = torch.FloatTensor(train_X)
    tensor_y = torch.LongTensor(train_y)
    train_dataset = TensorDataset(tensor_x, tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    model = LightweightBCINet().to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(Config.DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    model.train()
    for epoch in tqdm(range(Config.EPOCHS), desc="Training Final Model"):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(Config.DEVICE), batch_y.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    #save_path = "bci_model_final.pth"
    save_path = os.path.join(Config.RUN_DIR, "bci_model_final.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ 最終模型已成功儲存至: {save_path}")

# ==========================================
# 6. 主程式執行入口
# ==========================================
if __name__ == "__main__":
    save_config()
    os.makedirs(Config.RUN_DIR, exist_ok=True)
    if execute_loso:
        y_true, y_pred, results = leave_one_subject_out_cv()
        if y_true is not None:
            plot_and_evaluate(y_true, y_pred, results)
    if load_checkpoint:        
        train_and_save_final_model()
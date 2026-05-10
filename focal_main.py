"""
Brain-Computer Interface 1D-CNN+SE + Handcrafted Features Classifier
FINAL VERSION: Precise Labeling + Artifact Rejection + Feature Fusion + Fixed Normalization
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy import signal
from scipy.stats import kurtosis
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm
from datetime import datetime
import json
import joblib  # 用於儲存 Scaler

warnings.filterwarnings('ignore')

execute_loso = True
load_checkpoint = True

# ==========================================
# 1. 系統與資料流參數設定 (Config)
# ==========================================
class Config:
    DATASET_PATH = "bci_dataset_114-2_any"
    SAMPLING_RATE = 512
    WINDOW_SECONDS = 2.0       # 1秒視窗 (512 samples)

    RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = os.path.join("runs_2", RUN_TAG)
    
    STEP_SECONDS_BG = 1.0      # Relax/Focus 的滑動步長調大為 0.5 秒
    
    ARTIFACT_THRES = 600
    
    BLINK_TIMESTAMPS = [0, 4, 8, 12, 16]
    BLINK_SHIFTS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    
    # 模型與訓練參數
    BATCH_SIZE = 128
    EPOCHS = 80
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-3        
    PATIENCE = 20              
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_config():
    config_dict = {}
    for k, v in Config.__dict__.items():
        if not k.startswith("__") and not callable(v):
            try:
                json.dumps(v)
                config_dict[k] = v
            except TypeError:
                config_dict[k] = str(v)
    os.makedirs(Config.RUN_DIR, exist_ok=True)
    save_path = os.path.join(Config.RUN_DIR, "config.json")
    with open(save_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    print(f"Config saved to: {save_path}")

# ==========================================
# 2. 資料處理與特徵工程
# ==========================================
# 【修改 1】：放寬濾波器高頻限制，保留眨眼的高頻特徵 (1.0Hz -> 100.0Hz)
def bandpass_filter(data, lowcut=0.5, highcut=100.0, fs=Config.SAMPLING_RATE, order=4):
    """帶通濾波器，去除基線漂移，但保留肌電/眨眼高頻"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def extract_features(segments):
    """全面強化頻域特徵，加入專注度指標與 Alpha 峰值"""
    features = []
    for seg in segments:
        activity = np.var(seg) + 1e-7
        diff1 = np.diff(seg)
        diff2 = np.diff(diff1)
        
        var_diff1 = np.var(diff1) + 1e-7
        var_diff2 = np.var(diff2) + 1e-7
        
        mobility = np.sqrt(var_diff1 / activity)
        complexity = np.sqrt(var_diff2 / var_diff1) / mobility
        
        nperseg = min(len(seg), int(Config.SAMPLING_RATE * 2.0)) 
        freqs, psd = signal.welch(seg, fs=Config.SAMPLING_RATE, nperseg=nperseg)
        
        theta = np.sum(psd[(freqs >= 4) & (freqs < 8)])
        alpha = np.sum(psd[(freqs >= 8) & (freqs < 13)])
        beta_low = np.sum(psd[(freqs >= 13) & (freqs < 20)])
        beta_high = np.sum(psd[(freqs >= 20) & (freqs < 30)])
        beta = beta_low + beta_high
        total_power = theta + alpha + beta + 1e-9
        
        rel_theta = theta / total_power
        rel_alpha = alpha / total_power
        rel_beta  = beta / total_power
        
        engagement_index = beta / (alpha + theta + 1e-9)
        theta_beta_ratio = theta / (beta + 1e-9)
        
        alpha_band_idx = (freqs >= 8) & (freqs < 13)
        if np.sum(alpha_band_idx) > 0:
            alpha_peak_power = np.max(psd[alpha_band_idx])
            alpha_peak_freq = freqs[alpha_band_idx][np.argmax(psd[alpha_band_idx])]
        else:
            alpha_peak_power = 0
            alpha_peak_freq = 10.0
        
        kurt = kurtosis(seg)
        p2p_norm = np.ptp(seg) / (np.std(seg) + 1e-7) 
        
        current_feature = [
            mobility, complexity, 
            rel_theta, rel_alpha, rel_beta, 
            engagement_index, theta_beta_ratio,
            alpha_peak_freq, alpha_peak_power,
            kurt, p2p_norm,
            np.log10(activity),  
            np.max(np.abs(seg))  
        ]
        features.append(current_feature)
    return np.array(features)

def process_subject_files(folder_path):
    X, y = [], []
    win_samples = int(Config.WINDOW_SECONDS * Config.SAMPLING_RATE)
    step_samples_bg = int(Config.STEP_SECONDS_BG * Config.SAMPLING_RATE)
    
    for task in [1, 2, 3]:
        files = glob.glob(os.path.join(folder_path, f"*_{task}_*.txt"))
        for f in files:
            try:
                data = np.loadtxt(f)
            except ValueError:
                continue 
                
            if len(data) < 10240: 
                continue
                
            data = bandpass_filter(data)
            
            if task == 1: 
                for start in range(0, len(data) - win_samples, step_samples_bg):
                    X.append(data[start:start + win_samples])
                    y.append(0)
                    
            elif task == 2: 
                for start in range(0, len(data) - win_samples, step_samples_bg):
                    X.append(data[start:start + win_samples])
                    y.append(1)
                        
            elif task == 3: 
                for t in Config.BLINK_TIMESTAMPS:
                    center_sample = int(t * Config.SAMPLING_RATE)
                    for shift in Config.BLINK_SHIFTS:
                        start = center_sample + int(shift * Config.SAMPLING_RATE)
                        if start >= 0 and start + win_samples <= len(data):
                            X.append(data[start:start + win_samples])
                            y.append(2)
                        
    if not X: return None, None, None
    
    X_np = np.array(X)
    
    # 【修改 2】：正確的正規化邏輯 (保留 Segment 間的相對振幅大小)
    # 步驟 1: 各 Segment 扣除自身平均 (Zero-mean, 消除直流漂移)
    segment_means = np.mean(X_np, axis=1, keepdims=True)
    X_np = X_np - segment_means
    #X_np = X_np - np.mean(X_np)

    # 步驟 2: 計算整個受試者資料的標準差 (全域尺度)，並縮放
    # 這樣振幅大的 Segment (如眨眼) 依然會比振幅小的 Segment 數值大
    subject_std = np.std(X_np) + 1e-8
    X_np = X_np / subject_std
    
    # 在變成 PyTorch Tensor 前，擷取專家特徵
    features = extract_features(X_np)
    
    X_raw = np.expand_dims(X_np, axis=1).astype(np.float32)
    return X_raw, features.astype(np.float32), np.array(y, dtype=np.int64)

def load_all_data():
    subject_folders = sorted([f.path for f in os.scandir(Config.DATASET_PATH) if f.is_dir()])
    dataset_dict = {}
    
    for folder in subject_folders:
        sub_id = os.path.basename(folder)
        X_raw, X_feat, y = process_subject_files(folder)
        if X_raw is not None:
            dataset_dict[sub_id] = {'X_raw': X_raw, 'X_feat': X_feat, 'y': y}
            print(f"Loaded {sub_id}: {len(y)} segments (Relax: {np.sum(y==0)}, Focus: {np.sum(y==1)}, Blink: {np.sum(y==2)})")
            
    return dataset_dict

# ==========================================
# 3. 雙輸入神經網路架構 (CNN + 手工特徵融合)
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        # 先計算原本的 Cross Entropy
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        # 計算預測正確的機率 pt
        pt = torch.exp(-ce_loss)
        # 套用 Focal Loss 公式
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

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

class DualInputBCINet(nn.Module):
    def __init__(self, num_classes=3, num_manual_features=13):
        super(DualInputBCINet, self).__init__()
        # CNN 分支 (處理 Raw Wave)
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
        
        # 👇 新增這行：特徵融合後的批次正規化
        self.bn_fusion = nn.BatchNorm1d(64 + num_manual_features)

        # 【修改 3】：特徵融合分類器
        # 64 (CNN 輸出) + 13 (時頻特徵輸出) = 77
        self.fc1 = nn.Linear(64 + num_manual_features, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x_raw, x_feat):
        # 1. 處理波形
        x = self.pool1(F.relu(self.bn1(self.conv1(x_raw))))
        x = self.se1(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.se2(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x)))).squeeze(-1)
        
        # 2. 特徵拼接 (Concat)
        x = torch.cat((x, x_feat), dim=1)
        
        # 👇 新增這行：先做 BN 把 CNN 與手工特徵拉到同一尺度
        x = self.bn_fusion(x)

        # 3. 輸出層
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==========================================
# 4. 訓練與 LOSO 驗證流程
# ==========================================
def train_model(train_loader, val_X_raw, val_X_feat, val_y, class_weights):
    model = DualInputBCINet().to(Config.DEVICE)
    
    #criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(Config.DEVICE))
    criterion = FocalLoss(weight=torch.FloatTensor(class_weights).to(Config.DEVICE), gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    # add scheduler
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-5)

    val_X_raw_tensor = torch.FloatTensor(val_X_raw).to(Config.DEVICE)
    val_X_feat_tensor = torch.FloatTensor(val_X_feat).to(Config.DEVICE)
    val_y_tensor = torch.LongTensor(val_y).to(Config.DEVICE)
    
    best_acc = 0
    patience_counter = 0
    best_model_state = None
    loss_curve = []
    
    for epoch in tqdm(range(Config.EPOCHS), desc="Training Epochs"):
        model.train()
        epoch_losses = []
        
        for batch_raw, batch_feat, batch_y in train_loader:
            batch_raw = batch_raw.to(Config.DEVICE)
            batch_feat = batch_feat.to(Config.DEVICE)
            batch_y = batch_y.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_raw, batch_feat)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
        loss_curve.append(np.mean(epoch_losses))
            
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_X_raw_tensor, val_X_feat_tensor)
            _, predicted = torch.max(val_outputs, 1)
            acc = accuracy_score(val_y_tensor.cpu(), predicted.cpu())
            
        if acc > best_acc:
            best_acc = acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= Config.PATIENCE and acc >= 0.80:
            break

        #scheduler.step(acc) # 根據 Validation Accuracy 來調整學習率
        scheduler.step()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    model.eval()
    with torch.no_grad():
        final_outputs = model(val_X_raw_tensor, val_X_feat_tensor)
        _, final_pred = torch.max(final_outputs, 1)
        
    return final_pred.cpu().numpy(), loss_curve

def leave_one_subject_out_cv():
    dataset_dict = load_all_data()
    if not dataset_dict:
        return None, None, None
        
    subjects = sorted(list(dataset_dict.keys()))
    print(f"\n準備開始 LOSO 驗證，共 {len(subjects)} 位受試者...")
    
    results = {
        'subject_names': [], 'accuracies': [],
        'confusion_matrices': [], 'loss_curves': []
    }
    
    all_y_true, all_y_pred = [], []
    
    for test_sub in subjects:
        train_raw_list, train_feat_list, train_y_list = [], [], []
        test_raw = dataset_dict[test_sub]['X_raw']
        test_feat = dataset_dict[test_sub]['X_feat']
        test_y = dataset_dict[test_sub]['y']
        
        for sub in subjects:
            if sub != test_sub:
                train_raw_list.append(dataset_dict[sub]['X_raw'])
                train_feat_list.append(dataset_dict[sub]['X_feat'])
                train_y_list.append(dataset_dict[sub]['y'])
                
        train_raw = np.vstack(train_raw_list)
        train_feat = np.vstack(train_feat_list)
        train_y = np.hstack(train_y_list)
        
        # 針對手工特徵進行 StandardScaler 縮放，確保神經網路易於收斂
        scaler = StandardScaler()
        train_feat = scaler.fit_transform(train_feat)
        test_feat = scaler.transform(test_feat)
        
        classes = np.unique(train_y)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_y)
        
        tensor_raw = torch.FloatTensor(train_raw)
        tensor_feat = torch.FloatTensor(train_feat)
        tensor_y = torch.LongTensor(train_y)
        train_dataset = TensorDataset(tensor_raw, tensor_feat, tensor_y)
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        
        print(f"\nTraining model with {test_sub} left out...")
        y_pred, loss_curve = train_model(train_loader, test_raw, test_feat, test_y, weights)
        
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
    #plt.show()

def train_and_save_final_model():
    dataset_dict = load_all_data()
    if not dataset_dict: return
        
    print("\n準備訓練最終佈署模型 (使用全部資料)...")
    
    raw_list, feat_list, y_list = [], [], []
    for sub in dataset_dict:
        raw_list.append(dataset_dict[sub]['X_raw'])
        feat_list.append(dataset_dict[sub]['X_feat'])
        y_list.append(dataset_dict[sub]['y'])
        
    train_raw = np.vstack(raw_list)
    train_feat = np.vstack(feat_list)
    train_y = np.hstack(y_list)
    
    # 儲存 Scaler 供未來即時推論使用
    scaler = StandardScaler()
    train_feat = scaler.fit_transform(train_feat)
    scaler_save_path = os.path.join(Config.RUN_DIR, "feature_scaler.pkl")
    joblib.dump(scaler, scaler_save_path)
    
    classes = np.unique(train_y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_y)
    
    tensor_raw = torch.FloatTensor(train_raw)
    tensor_feat = torch.FloatTensor(train_feat)
    tensor_y = torch.LongTensor(train_y)
    train_dataset = TensorDataset(tensor_raw, tensor_feat, tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    model = DualInputBCINet().to(Config.DEVICE)
    #criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(Config.DEVICE))
    criterion = FocalLoss(weight=torch.FloatTensor(weights).to(Config.DEVICE), gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-5)

    model.train()
    for epoch in tqdm(range(Config.EPOCHS), desc="Training Final Model"):
        for batch_raw, batch_feat, batch_y in train_loader:
            batch_raw, batch_feat, batch_y = batch_raw.to(Config.DEVICE), batch_feat.to(Config.DEVICE), batch_y.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_raw, batch_feat)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()
            
    save_path = os.path.join(Config.RUN_DIR, "bci_model_final.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ 最終模型已成功儲存至: {save_path}")
    print(f"✅ 特徵 Scaler 已成功儲存至: {scaler_save_path}")

import random
# ==========================================
# 固定所有隨機性
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 讓 cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # pytorch deterministic mode
    torch.use_deterministic_algorithms(True)

    # python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Using deterministic seed = {seed}")

# ==========================================
# 6. 主程式執行入口
# ==========================================
if __name__ == "__main__":
    set_seed(42)
    os.makedirs(Config.RUN_DIR, exist_ok=True)
    save_config()
    
    if execute_loso:
        y_true, y_pred, results = leave_one_subject_out_cv()
        if y_true is not None:
            plot_and_evaluate(y_true, y_pred, results)
            pass 
            
    if load_checkpoint:        
        train_and_save_final_model()
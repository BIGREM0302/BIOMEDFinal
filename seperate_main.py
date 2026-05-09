"""
Brain-Computer Interface Two-Stage Classifier (Blink vs Mental State)
BASED ON: Dual Temporal Scale + Feature Engineering + Machine Learning
"""

import os
import glob
import numpy as np
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==========================================
# 1. 系統與資料流參數設定 (Config)
# ==========================================
class Config:
    DATASET_PATH = "bci_dataset_114-2_any"
    SAMPLING_RATE = 512
    
    # --- Stage 1: Blink Detector 參數 ---
    BLINK_WINDOW_SEC = 0.5      # 0.5秒 (256 samples)
    BLINK_HOP_SEC = 0.125       # 滑動 0.125秒 (64 samples)
    BLINK_TIMESTAMPS = [0, 4, 8, 12, 16] # 刻意眨眼時間點
    BLINK_MARGIN = 0.25         # ±250ms 視為 Positive Blink
    
    # --- Stage 2: Mental State (Relax/Focus) 參數 ---
    STATE_WINDOW_SEC = 4.0      # 4秒 (2048 samples)
    STATE_HOP_SEC = 0.5         # 滑動 0.5秒 (256 samples)
    ARTIFACT_THRES = 400        # 振幅超過此值視為含有眨眼等雜訊，予以排除

# ==========================================
# 2. 濾波與特徵萃取 (Feature Engineering)
# ==========================================
def bandpass_filter(data, lowcut=1.0, highcut=40.0, fs=Config.SAMPLING_RATE, order=4):
    """基礎帶通濾波器"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def extract_blink_features(segment):
    """Stage 1 特徵：抓取瞬時強烈變化"""
    peak_amp = np.max(np.abs(segment))
    variance = np.var(segment)
    return [peak_amp, variance]

def extract_state_features(segment, fs=Config.SAMPLING_RATE):
    """Stage 2 特徵：頻域能量 (FFT Band Power)"""
    # Welch's method 計算 PSD (Power Spectral Density)
    freqs, psd = signal.welch(segment, fs, nperseg=fs) # 1秒解析度
    
    theta_power = np.sum(psd[(freqs >= 4) & (freqs < 8)])
    alpha_power = np.sum(psd[(freqs >= 8) & (freqs < 13)])
    beta_power = np.sum(psd[(freqs >= 13) & (freqs < 30)])
    
    total_power = theta_power + alpha_power + beta_power + 1e-8
    
    # 取相對能量與 Alpha/Beta Ratio
    rel_alpha = alpha_power / total_power
    rel_beta = beta_power / total_power
    alpha_beta_ratio = alpha_power / (beta_power + 1e-8)
    
    return [rel_alpha, rel_beta, alpha_beta_ratio, theta_power]

# ==========================================
# 3. 資料處理與切割 (Segmentation & Labeling)
# ==========================================
def process_subject_files(folder_path):
    """將受試者資料分別切割給 Blink Detector 與 State Classifier"""
    blink_X, blink_y = [], []
    state_X, state_y = [], []
    
    fs = Config.SAMPLING_RATE
    
    # 處理所有檔案
    for task in [1, 2, 3]:
        files = glob.glob(os.path.join(folder_path, f"*_{task}_*.txt"))
        for f in files:
            try:
                raw_data = np.loadtxt(f)
            except ValueError:
                continue
            if len(raw_data) < fs * 20: continue # 確保長度足夠
            
            data = bandpass_filter(raw_data)
            
            # --------------------------------------------------
            # 資料集 A: 給 Stage 1 (Blink Detector)
            # 使用 0.5s window, 任務3 抓正負樣本，任務1/2 皆為負樣本
            # --------------------------------------------------
            win_b = int(Config.BLINK_WINDOW_SEC * fs)
            hop_b = int(Config.BLINK_HOP_SEC * fs)
            
            for start in range(0, len(data) - win_b, hop_b):
                segment = data[start:start + win_b]
                features = extract_blink_features(segment)
                
                if task == 3:
                    # 計算這個 window 的中心時間點
                    center_time = (start + win_b / 2) / fs
                    # 判斷是否在眨眼的時間容忍區間內 (±250ms)
                    is_blink = any(abs(center_time - t) <= Config.BLINK_MARGIN for t in Config.BLINK_TIMESTAMPS)
                    blink_X.append(features)
                    blink_y.append(1 if is_blink else 0)
                else:
                    # Task 1 & 2 裡面理論上沒有「刻意眨眼」
                    blink_X.append(features)
                    blink_y.append(0)

            # --------------------------------------------------
            # 資料集 B: 給 Stage 2 (Relax vs Focus)
            # 使用 4.0s window, 僅限任務 1 與 2，並排除雜訊
            # --------------------------------------------------
            if task in [1, 2]:
                win_s = int(Config.STATE_WINDOW_SEC * fs)
                hop_s = int(Config.STATE_HOP_SEC * fs)
                
                for start in range(0, len(data) - win_s, hop_s):
                    segment = data[start:start + win_s]
                    
                    # 排除振幅過大(如不經意眨眼或肌肉活動)的雜訊段落
                    if np.max(np.abs(segment)) < Config.ARTIFACT_THRES:
                        features = extract_state_features(segment)
                        state_X.append(features)
                        state_y.append(0 if task == 1 else 1) # 0:Relax, 1:Focus
                        
    return np.array(blink_X), np.array(blink_y), np.array(state_X), np.array(state_y)

def load_all_data():
    subject_folders = sorted([f.path for f in os.scandir(Config.DATASET_PATH) if f.is_dir()])
    dataset_dict = {}
    
    for folder in subject_folders:
        sub_id = os.path.basename(folder)
        bx, by, sx, sy = process_subject_files(folder)
        if len(bx) > 0 and len(sx) > 0:
            dataset_dict[sub_id] = {'blink_X': bx, 'blink_y': by, 'state_X': sx, 'state_y': sy}
            print(f"[{sub_id}] Blink data: {len(by)} | State data: {len(sy)} (Relax:{np.sum(sy==0)} Focus:{np.sum(sy==1)})")
            
    return dataset_dict

# ==========================================
# 4. 訓練與 LOSO 驗證流程 (Two-Stage)
# ==========================================
def leave_one_subject_out_cv():
    dataset_dict = load_all_data()
    if not dataset_dict:
        print("未找到資料！")
        return
        
    subjects = sorted(list(dataset_dict.keys()))
    print(f"\n準備開始 LOSO 驗證，共 {len(subjects)} 位受試者...")
    
    # 紀錄兩階段的真實與預測結果
    res_blink = {'true': [], 'pred': []}
    res_state = {'true': [], 'pred': []}
    
    for test_sub in subjects:
        print(f"Testing on Subject {test_sub}...")
        
        # --- 組合 Training Data ---
        train_bx, train_by, train_sx, train_sy = [], [], [], []
        for sub in subjects:
            if sub != test_sub:
                train_bx.append(dataset_dict[sub]['blink_X'])
                train_by.append(dataset_dict[sub]['blink_y'])
                train_sx.append(dataset_dict[sub]['state_X'])
                train_sy.append(dataset_dict[sub]['state_y'])
                
        X_train_b = np.vstack(train_bx)
        y_train_b = np.hstack(train_by)
        X_train_s = np.vstack(train_sx)
        y_train_s = np.hstack(train_sy)
        
        # --- 載入 Testing Data ---
        X_test_b = dataset_dict[test_sub]['blink_X']
        y_test_b = dataset_dict[test_sub]['blink_y']
        X_test_s = dataset_dict[test_sub]['state_X']
        y_test_s = dataset_dict[test_sub]['state_y']
        
        # --- Stage 1: 訓練 Blink Detector (Random Forest) ---
        clf_blink = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)
        clf_blink.fit(X_train_b, y_train_b)
        pred_b = clf_blink.predict(X_test_b)
        
        res_blink['true'].extend(y_test_b)
        res_blink['pred'].extend(pred_b)
        
        # --- Stage 2: 訓練 Mental State Classifier (SVM) ---
        clf_state = SVC(kernel='rbf', class_weight='balanced', random_state=42)
        clf_state.fit(X_train_s, y_train_s)
        pred_s = clf_state.predict(X_test_s)
        
        res_state['true'].extend(y_test_s)
        res_state['pred'].extend(pred_s)

    return res_blink, res_state

# ==========================================
# 5. 結果評估與視覺化
# ==========================================
def evaluate_results(res_blink, res_state):
    print("\n" + "="*50)
    print("STAGE 1: Blink Detector Performance (0.5s Window)")
    print("="*50)
    print(classification_report(res_blink['true'], res_blink['pred'], target_names=['Normal', 'Intentional Blink']))
    
    print("\n" + "="*50)
    print("STAGE 2: Mental State Performance (4.0s Window)")
    print("="*50)
    print(classification_report(res_state['true'], res_state['pred'], target_names=['Relax', 'Focus']))

# ==========================================
# 6. 即時系統流程模擬 (Real-time Pipeline)
# ==========================================
class RealtimeSystemMockup:
    """展示如何將兩階段模型套用在即時 EEG 資料流 (每 0.5 秒更新一次預測)"""
    def __init__(self, blink_model, state_model):
        self.blink_model = blink_model
        self.state_model = state_model
        
        self.fs = Config.SAMPLING_RATE
        self.buffer_4s = np.zeros(int(4.0 * self.fs)) # 維持一個 4秒的長 buffer
        
    def stream_update(self, new_0_5s_data):
        """每收滿 0.5 秒 (256點) 資料就呼叫一次此函數"""
        
        # 1. 濾波
        filtered_new = bandpass_filter(new_0_5s_data)
        
        # 2. Stage 1: 先檢查這 0.5 秒是不是刻意眨眼
        blink_feat = extract_blink_features(filtered_new)
        is_blink = self.blink_model.predict([blink_feat])[0]
        
        if is_blink == 1:
            return "Event: INTENTIONAL BLINK!"
            
        # 3. Stage 2: 如果不是刻意眨眼，更新 4秒 buffer
        # 將舊資料往前推，新資料補在尾端
        shift_len = len(filtered_new)
        self.buffer_4s = np.roll(self.buffer_4s, -shift_len)
        self.buffer_4s[-shift_len:] = filtered_new
        
        # 提取 4秒特徵
        state_feat = extract_state_features(self.buffer_4s)
        state_pred = self.state_model.predict([state_feat])[0]
        
        return "State: FOCUS" if state_pred == 1 else "State: RELAX"

if __name__ == "__main__":
    res_blink, res_state = leave_one_subject_out_cv()
    if res_blink:
        evaluate_results(res_blink, res_state)
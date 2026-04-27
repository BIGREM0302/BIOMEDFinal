"""
Brain-Computer Interface MLP Classifier
FINAL VERSION: DYNAMIC SEGMENTATION + HJORTH COMPLEXITY & ENHANCED FREQUENCY FEATURES
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from scipy import signal
from scipy.stats import kurtosis, skew
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# Parameter Settings
class Config:
    DATASET_PATH = "bci_dataset_114-2"
    SKIP_SECONDS = 2.0                 # 捨棄每回合開頭前 2 秒
    
    # === 策略 A：針對連續狀態 (Relax / Focus) ===
    RF_SEG_LEN = 4.0                   # 窗口大一點，頻譜解析度才高
    # 【修改 1】針對不同任務給予不同的 Overlap，解決資料不平衡！
    RELAX_OVERLAP = 0.7               # Relax 切密一點，補足資料量
    FOCUS_OVERLAP = 0.7                # Focus 維持 0.7
    
    # 【修改 1】門檻統一放寬，保留更多原始資料，防止 Relax 被過度刪除
    MAX_THRES = 1500                 
    RELAX_THRES = 1500
    FOCUS_THRES = 1200

    # === 策略 B：針對瞬間狀態 (Blink) ===
    BLINK_SEG_LEN = 1.5                # 窗口縮小，聚焦眨眼瞬間，避免被背景稀釋
    BLINK_OVERLAP = 0.0                # 重疊率 0，不重複計算同一個眨眼
    BLINK_MIN_THRES = 500              # 必須有大於 500 的突波才承認是眨眼
    
    # MLP model parameters
    HIDDEN_LAYERS = (64, 32)           
    MAX_ITER = 200                     
    LEARNING_RATE = 0.005              
    ALPHA = 0.05                       # 提高正規化強度，防止模型死背特徵
    ACTIVATION = 'relu'                
    SOLVER = 'adam'                    
    BATCH_SIZE = 128                   
    EARLY_STOPPING = True              
    VALIDATION_FRACTION = 0.1
    N_ITER_NO_CHANGE = 15
    SAMPLING_RATE = 512                
    FEATURE_SELECTION = True
    N_FEATURES_SELECT = 10             # 【修改 1】特徵變多了，選前 15 個最強的
    RANDOM_STATE = 42

def create_segments(data, segment_length_samples, overlap_samples, task_type):
    """根據不同任務類型，執行不同的切割與過濾策略"""
    skip_samples = int(Config.SKIP_SECONDS * Config.SAMPLING_RATE)
    if len(data) > skip_samples:
        data = data[skip_samples:]
    else:
        return []
        
    if len(data) < segment_length_samples:
        return []
    
    segments = []
    start = 0
    step = segment_length_samples - overlap_samples
    
    nyq = 0.5 * Config.SAMPLING_RATE
    low, high = 0.5 / nyq, 45.0 / nyq
    b, a = signal.butter(4, [low, high], btype='band')

    while start + segment_length_samples <= len(data):
        segment = data[start:start + segment_length_samples]
        segment_filtered = signal.filtfilt(b, a, segment)
        peak_amp = np.max(np.abs(segment_filtered))
        
        # === 核心邏輯：依照任務進行智能過濾 ===
        if task_type == 1: # Relax
            # 【修改 1】統一振幅門檻
            if peak_amp > Config.RELAX_THRES: 
                start += step
                continue 
        elif task_type == 2:
            if peak_amp > Config.FOCUS_THRES:
                start += step
                continue
        elif task_type == 3:    # Blink
            # 如果這個小視窗內沒有出現足夠大的突波，代表它切到了「沒眨眼」的空白期
            if peak_amp < Config.BLINK_MIN_THRES:
                start += step
                continue # 沒有眨眼的片段直接丟棄，防止標籤污染
            
        segments.append(segment_filtered)
        start += step
        
    return segments

def extract_features(segments):
    """【修改 2】全面強化頻域特徵，加入專注度指標與 Alpha 峰值"""
    features = []
    for seg in segments:
        # 1. Hjorth Parameters (Activity, Mobility, Complexity)
        activity = np.var(seg) + 1e-7
        diff1 = np.diff(seg)
        diff2 = np.diff(diff1)
        
        var_diff1 = np.var(diff1) + 1e-7
        var_diff2 = np.var(diff2) + 1e-7
        
        mobility = np.sqrt(var_diff1 / activity)
        complexity = np.sqrt(var_diff2 / var_diff1) / mobility
        
        # 2. 頻域特徵 (提高解析度至 0.5Hz)
        nperseg = min(len(seg), int(Config.SAMPLING_RATE * 2.0)) 
        freqs, psd = signal.welch(seg, fs=Config.SAMPLING_RATE, nperseg=nperseg)
        
        theta = np.sum(psd[(freqs >= 4) & (freqs < 8)])
        alpha = np.sum(psd[(freqs >= 8) & (freqs < 13)])
        beta_low = np.sum(psd[(freqs >= 13) & (freqs < 20)])
        beta_high = np.sum(psd[(freqs >= 20) & (freqs < 30)])
        beta = beta_low + beta_high
        total_power = theta + alpha + beta + 1e-9
        
        # 相對能量比例
        rel_theta = theta / total_power
        rel_alpha = alpha / total_power
        rel_beta  = beta / total_power
        
        # 專注度指標 (Engagement Index) 與 Theta/Beta 比例
        engagement_index = beta / (alpha + theta + 1e-9)
        theta_beta_ratio = theta / (beta + 1e-9)
        
        # 尋找 Alpha 波段的最大峰值特徵
        alpha_band_idx = (freqs >= 8) & (freqs < 13)
        if np.sum(alpha_band_idx) > 0:
            alpha_peak_power = np.max(psd[alpha_band_idx])
            alpha_peak_freq = freqs[alpha_band_idx][np.argmax(psd[alpha_band_idx])]
        else:
            alpha_peak_power = 0
            alpha_peak_freq = 10.0 # 預設值
        
        # 3. 輔助特徵 
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

def load_all_subjects():
    all_features, all_labels, all_subjects = [], [], []
    if not os.path.exists(Config.DATASET_PATH): return None, None, None
    subject_folders = sorted([f.path for f in os.scandir(Config.DATASET_PATH) if f.is_dir()])
    
    rf_seg_len = int(Config.RF_SEG_LEN * Config.SAMPLING_RATE)
    blk_seg_len = int(Config.BLINK_SEG_LEN * Config.SAMPLING_RATE)
    blk_overlap = int(blk_seg_len * Config.BLINK_OVERLAP)

    for folder in subject_folders:
        sub_id = os.path.basename(folder)
        sub_segs = {1: [], 2: [], 3: []}
        
        for task in [1, 2, 3]:
            files = glob.glob(os.path.join(folder, f"*_{task}_*.txt"))
            for f in files:
                try:
                    with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                        lines = file.readlines()
                        clean_data = []
                        for line in lines:
                            val = line.strip()
                            if val:
                                try: clean_data.append(float(val))
                                except ValueError: pass 
                    data = np.array(clean_data)
                    
                    # 【修改 1】根據任務決定切割的重疊率
                    if task == 1:
                        overlap = int(rf_seg_len * Config.RELAX_OVERLAP)
                        segs = create_segments(data, rf_seg_len, overlap, task)
                    elif task == 2:
                        overlap = int(rf_seg_len * Config.FOCUS_OVERLAP)
                        segs = create_segments(data, rf_seg_len, overlap, task)
                    else:
                        segs = create_segments(data, blk_seg_len, blk_overlap, task)
                        
                    sub_segs[task].extend(segs)
                        
                except Exception as e:
                    continue
        
        if not (sub_segs[1] and sub_segs[2] and sub_segs[3]): 
            print(f"Warning: Subject {sub_id} missing valid task data. Skipping.")
            continue

        f1 = extract_features(sub_segs[1])
        f2 = extract_features(sub_segs[2])
        f3 = extract_features(sub_segs[3])
        
        sub_feat = np.vstack([f1, f2, f3])
        sub_feat = StandardScaler().fit_transform(sub_feat) 
        
        sub_lab = np.hstack([np.zeros(len(f1)), np.ones(len(f2)), np.full(len(f3), 2)])
        all_features.append(sub_feat)
        all_labels.append(sub_lab)
        all_subjects.extend([sub_id] * len(sub_lab))
        
        print(f" - {sub_id}: Loaded Relax({len(f1)}), Focus({len(f2)}), Blink({len(f3)}) segments")
    
    if not all_features: return None, None, None
    return np.vstack(all_features), np.hstack(all_labels), all_subjects

class EnhancedBCIClassifier:
    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=Config.HIDDEN_LAYERS, 
            max_iter=Config.MAX_ITER,
            learning_rate_init=Config.LEARNING_RATE, 
            alpha=Config.ALPHA,
            activation=Config.ACTIVATION, 
            solver=Config.SOLVER, 
            batch_size=Config.BATCH_SIZE,
            early_stopping=Config.EARLY_STOPPING, 
            validation_fraction=Config.VALIDATION_FRACTION,
            n_iter_no_change=Config.N_ITER_NO_CHANGE,
            random_state=Config.RANDOM_STATE,
            verbose=False
        )
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=Config.N_FEATURES_SELECT) if Config.FEATURE_SELECTION else None
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        X_selected = self.feature_selector.fit_transform(X_scaled, y) if self.feature_selector else X_scaled
        self.model.fit(X_selected, y)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled) if self.feature_selector else X_scaled
        
        # 1. 取得三個類別的機率 [Relax(0), Focus(1), Blink(2)]
        probs = self.model.predict_proba(X_selected)
        
        predictions = np.zeros(len(X), dtype=int)
        
        # 2. 客製化門檻邏輯
        for i in range(len(X)):
            # 只要 Relax 的機率超過 0.35 (不用等到 0.5 或最高)，就判定為 Relax
            if probs[i, 0] > 0.6:  
                predictions[i] = 0
            else:
                # 剩下的再讓 Focus 和 Blink 去比誰機率高
                # np.argmax(probs[i, 1:]) 會回傳 0(對應Focus) 或 1(對應Blink)，所以要 +1
                predictions[i] = np.argmax(probs[i, 1:]) + 1
                
        # 3. 平滑化
        if len(predictions) > 5:
            return signal.medfilt(predictions, kernel_size=11) 
        return predictions

def leave_one_subject_out_validation():
    print("\nStarting Leave-One-Subject-Out (LOSO) Cross-Validation...")
    X, y, subjects = load_all_subjects()
    if X is None: return None
        
    unique_subjects = sorted(list(set(subjects)))
    results = {'accuracies': [], 'confusion_matrices': [], 'loss_curves': [], 'subject_names': []}
    
    for test_sub in unique_subjects:
        train_mask = [s != test_sub for s in subjects]
        test_mask = [s == test_sub for s in subjects]
        
        clf = EnhancedBCIClassifier().fit(X[train_mask], y[train_mask])
        y_pred = clf.predict(X[test_mask])
        
        results['accuracies'].append(accuracy_score(y[test_mask], y_pred))
        results['confusion_matrices'].append(confusion_matrix(y[test_mask], y_pred, labels=[0, 1, 2]))
        results['loss_curves'].append(clf.model.loss_curve_)
        results['subject_names'].append(test_sub)
        
        print(f" -> {test_sub} Accuracy: {results['accuracies'][-1]:.3f}")
    return results

def plot_results(results):
    if results is None: return
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('BCI Classifier (Raw Data) - Group LOSO Results', fontsize=16)
    
    # 1. Accuracy distribution
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
    
    # 2. Overall confusion matrix
    total_cm = np.sum(results['confusion_matrices'], axis=0)
    sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Relax', 'Focus', 'Blink'], yticklabels=['Relax', 'Focus', 'Blink'], ax=axes[1])
    axes[1].set_title('Overall Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    # 3. Training loss curves
    valid_loss_curves = [lc for lc in results['loss_curves'] if len(lc) > 0]
    if valid_loss_curves:
        for i, loss_curve in enumerate(valid_loss_curves):
            axes[2].plot(loss_curve, alpha=0.7, label=subject_names[i])
        axes[2].set_title('Training Loss Curves')
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bci_results_raw_data.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("BCI EEG Classification - Group Evaluation")
    print("=" * 60)
    
    results = leave_one_subject_out_validation()
    if results is None:
        print("Validation failed! Please check your directory structure.")
        return
    
    mean_accuracy = np.mean(results['accuracies'])
    std_accuracy = np.std(results['accuracies'])
    
    print("\n" + "="*40)
    print(f"Overall Mean Accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
    
    total_cm = np.sum(results['confusion_matrices'], axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        relax_accuracy = total_cm[0, 0] / np.sum(total_cm[0, :]) if np.sum(total_cm[0, :]) > 0 else 0
        concentration_accuracy = total_cm[1, 1] / np.sum(total_cm[1, :]) if np.sum(total_cm[1, :]) > 0 else 0
        blink_accuracy = total_cm[2, 2] / np.sum(total_cm[2, :]) if np.sum(total_cm[2, :]) > 0 else 0
        relax_precision = total_cm[0, 0] / np.sum(total_cm[:, 0]) if np.sum(total_cm[:, 0]) > 0 else 0
        concentration_precision = total_cm[1, 1] / np.sum(total_cm[:, 1]) if np.sum(total_cm[:, 1]) > 0 else 0
        blink_precision = total_cm[2, 2] / np.sum(total_cm[:, 2]) if np.sum(total_cm[:, 2]) > 0 else 0

    print(f"\n[Relax Class]")
    print(f"  - Accuracy (Recall): {relax_accuracy:.3f} ({total_cm[0, 0]}/{np.sum(total_cm[0, :])})")
    print(f"  - Precision: {relax_precision:.3f} ({total_cm[0, 0]}/{np.sum(total_cm[:, 0])})")
    
    print(f"\n[Focus Class]")
    print(f"  - Accuracy (Recall): {concentration_accuracy:.3f} ({total_cm[1, 1]}/{np.sum(total_cm[1, :])})")
    print(f"  - Precision: {concentration_precision:.3f} ({total_cm[1, 1]}/{np.sum(total_cm[:, 1])})")
    
    print(f"\n[Blink Class]")
    print(f"  - Accuracy (Recall): {blink_accuracy:.3f} ({total_cm[2, 2]}/{np.sum(total_cm[2, :])})")
    print(f"  - Precision: {blink_precision:.3f} ({total_cm[2, 2]}/{np.sum(total_cm[:, 2])})")
    
    plot_results(results)
    print(f"\nResults saved to 'bci_results_raw_data.png'")

if __name__ == "__main__":
    main()
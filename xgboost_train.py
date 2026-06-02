"""
Brain-Computer Interface: XGBoost Ultimate Baseline
Features: XGBoost + Feature Selection/Addition + HPO (RandomizedSearch) + Subject Calibration (Personalization)
"""

import os
import glob
import numpy as np
import xgboost as xgb
from scipy import signal
from scipy.stats import kurtosis, skew
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import joblib

warnings.filterwarnings('ignore')

execute_loso = True
train_final_model = True

# ==========================================
# 1. Parameter Settings
# ==========================================
class Config:
    DATASET_PATH = "bci_dataset_114-2_any"  

    # exclude subjects with poor data quality (based on observing the plots)
    exclude_subjects = ["S06", "S14", "S09", "S05"]  

    SAMPLING_RATE = 512
    WINDOW_SECONDS = 1.0       
    RUN_TAG = datetime.now().strftime("XGB_%Y%m%d_%H%M%S")
    RUN_DIR = os.path.join("runs", RUN_TAG)
    STEP_SECONDS_BG = 0.5      
    BLINK_TIMESTAMPS = [0, 4, 8, 12, 16]
    BLINK_SHIFTS = [-0.3,  -0.2,  -0.1, 0.0, 0.1,  0.2,  0.3]
    
    # personal calibration ratio
    CALIBRATION_RATIO = 0.0   

FEATURE_NAMES = [
    'Mobility', 'Complexity', 
    'Rel Theta', 'Rel Beta',            
    'Engagement Idx', 'Theta/Beta Ratio',
    'Alpha Peak Pwr',                   
    'Kurtosis', 'Skewness',             
    'P2P Norm', 
    # 'Log Activity', 
    'Spectral Entropy',      
    'Zero Crossing Rate', 
    # 'Root Mean Square', 
    'Alpha/Beta Ratio', 'Rel Delta', 'Rel Gamma'
]

# ==========================================
# 2. Data Processing & Feature Extraction
# ==========================================
def bandpass_filter(data, lowcut=0.5, highcut=100.0, fs=Config.SAMPLING_RATE, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def extract_features(segments):
    features = []
    for seg in segments:
        # Hjorth Parameters
        activity = np.var(seg) + 1e-7
        diff1 = np.diff(seg)
        diff2 = np.diff(diff1)
        var_diff1 = np.var(diff1) + 1e-7
        var_diff2 = np.var(diff2) + 1e-7
        mobility = np.sqrt(var_diff1 / activity)
        complexity = np.sqrt(var_diff2 / var_diff1) / mobility
        
        # Frequency domain features
        nperseg = min(len(seg), int(Config.SAMPLING_RATE * 2.0)) 
        freqs, psd = signal.welch(seg, fs=Config.SAMPLING_RATE, nperseg=nperseg)
        
        theta = np.sum(psd[(freqs >= 4) & (freqs < 8)])
        alpha = np.sum(psd[(freqs >= 8) & (freqs < 13)])
        beta = np.sum(psd[(freqs >= 13) & (freqs < 30)])
        delta = np.sum(psd[(freqs >= 1) & (freqs < 4)])
        gamma = np.sum(psd[(freqs >= 30) & (freqs < 40)])
        total_power = theta + alpha + beta + delta + gamma + 1e-9
        
        rel_theta = theta / total_power
        rel_beta  = beta / total_power  

        alpha_beta_ratio = alpha / (beta + 1e-9) 
        rel_delta = delta / total_power
        rel_gamma = gamma / total_power
        
        engagement_index = beta / (alpha + theta + 1e-9)
        theta_beta_ratio = theta / (beta + 1e-9)
        
        alpha_band_idx = (freqs >= 8) & (freqs < 13)
        alpha_peak_power = np.max(psd[alpha_band_idx]) if np.sum(alpha_band_idx) > 0 else 0
        
        kurt = kurtosis(seg)
        skew_val = skew(seg) 
        p2p_norm = np.ptp(seg) / (np.std(seg) + 1e-7) 
        
        psd_norm = psd / (np.sum(psd) + 1e-9)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-9))

        # time domain features
        zcr = np.sum(np.diff(np.sign(seg)) != 0) / len(seg)

        rms = np.sqrt(np.mean(seg**2))
        
        current_feature = [
            mobility, complexity, 
            rel_theta, rel_beta, 
            engagement_index, theta_beta_ratio,
            alpha_peak_power,
            kurt, skew_val, p2p_norm,
            # np.log10(activity), 
            spectral_entropy, zcr, 
            # rms, 
            alpha_beta_ratio, rel_delta, rel_gamma
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
            if any(bad_id in f for bad_id in Config.exclude_subjects):
                #print(f"Data skipped: {f}")
                continue

            try:
                data = np.loadtxt(f)
            except ValueError:
                continue 
            if len(data) < 10240: continue
            data = bandpass_filter(data)
            
            if task == 1: 
                for start in range(0, len(data) - win_samples, step_samples_bg):
                    X.append(data[start:start + win_samples]), y.append(0)
            elif task == 2: 
                for start in range(0, len(data) - win_samples, step_samples_bg):
                    X.append(data[start:start + win_samples]), y.append(1)
            elif task == 3: 
                for t in Config.BLINK_TIMESTAMPS:
                    center_sample = int(t * Config.SAMPLING_RATE)
                    for shift in Config.BLINK_SHIFTS:
                        start = center_sample + int(shift * Config.SAMPLING_RATE)
                        if start >= 0 and start + win_samples <= len(data):
                            X.append(data[start:start + win_samples]), y.append(2)
                        
    if not X: return None, None
    X_np = np.array(X)
    X_np = (X_np - np.mean(X_np)) / (np.std(X_np) + 1e-8)
    return extract_features(X_np).astype(np.float32), np.array(y, dtype=np.int64)

def load_all_data():
    subject_folders = sorted([f.path for f in os.scandir(Config.DATASET_PATH) if f.is_dir()])
    dataset_dict = {}
    for folder in subject_folders:
        sub_id = os.path.basename(folder)
        X_feat, y = process_subject_files(folder)
        if X_feat is not None:
            dataset_dict[sub_id] = {'X_feat': X_feat, 'y': y}
            print(f"Loading {sub_id}: total segments of {len(y)} ")
    return dataset_dict

# ==========================================
# 3. XGBoost training with LOSO 
# ==========================================
def _leave_one_subject_out_cv():
    dataset_dict = load_all_data()
    if not dataset_dict:
        return None, None, None
        
    subjects = sorted(list(dataset_dict.keys()))
    print("\n準備開始 Pure LOSO 驗證（無個人化校正）...")
    
    results = {
        'subject_names': [],
        'accuracies': [],
        'confusion_matrices': [],
        'feature_importances': []
    }

    all_y_true, all_y_pred = [], []
    
    for test_sub in subjects:
        train_feat_list, train_y_list = [], []

        # 測試資料：完整的 test subject
        test_feat = dataset_dict[test_sub]['X_feat']
        test_y = dataset_dict[test_sub]['y']
        
        # 訓練資料：除了 test subject 以外的所有 subjects
        for sub in subjects:
            if sub != test_sub:
                train_feat_list.append(dataset_dict[sub]['X_feat'])
                train_y_list.append(dataset_dict[sub]['y'])
                
        train_feat = np.vstack(train_feat_list)
        train_y = np.hstack(train_y_list)
        
        scaler = StandardScaler()
        train_feat_scaled = scaler.fit_transform(train_feat)
        test_feat_scaled = scaler.transform(test_feat)
        
        sample_weights = compute_sample_weight(class_weight='balanced', y=train_y)
        
        print(f"Training XGBoost with {test_sub} left out (No Calibration)... ", end="")
        
        xgb_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            tree_method='hist',
            random_state=42,
            n_jobs=-1
        )
        
        xgb_model.fit(train_feat_scaled, train_y, sample_weight=sample_weights)
        y_pred = xgb_model.predict(test_feat_scaled)
        
        acc = accuracy_score(test_y, y_pred)

        results['subject_names'].append(test_sub)
        results['accuracies'].append(acc)
        results['confusion_matrices'].append(
            confusion_matrix(test_y, y_pred, labels=[0, 1, 2])
        )
        results['feature_importances'].append(xgb_model.feature_importances_)
        
        all_y_true.extend(test_y)
        all_y_pred.extend(y_pred)

        print(f"Test Accuracy: {acc:.4f}")
        
    return all_y_true, all_y_pred, results

def leave_one_subject_out_cv():
    dataset_dict = load_all_data()
    if not dataset_dict: return None, None, None
        
    subjects = sorted(list(dataset_dict.keys()))
    print(f"\n準備開始 LOSO 驗證 (包含 {Config.CALIBRATION_RATIO*100}% 個人化校正)...")
    
    results = {'subject_names': [], 'accuracies': [], 'confusion_matrices': [], 'feature_importances': []}
    all_y_true, all_y_pred = [], []
    
    for test_sub in subjects:
        train_feat_list, train_y_list = [], []
        
        # personal calibration
        test_feat_full = dataset_dict[test_sub]['X_feat']
        test_y_full = dataset_dict[test_sub]['y']
        
        calib_feat, test_feat, calib_y, test_y = train_test_split(
            test_feat_full, test_y_full, 
            test_size=(1.0 - Config.CALIBRATION_RATIO), 
            random_state=42, stratify=test_y_full
        )
        
        for sub in subjects:
            if sub != test_sub:
                train_feat_list.append(dataset_dict[sub]['X_feat'])
                train_y_list.append(dataset_dict[sub]['y'])
                
        train_feat = np.vstack(train_feat_list + [calib_feat])
        train_y = np.hstack(train_y_list + [calib_y])
        
        scaler = StandardScaler()
        train_feat_scaled = scaler.fit_transform(train_feat)
        test_feat_scaled = scaler.transform(test_feat)
        
        sample_weights = compute_sample_weight(class_weight='balanced', y=train_y)
        
        print(f"Training XGBoost with {test_sub} left out (Calibrated)... ", end="")
        
        xgb_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            tree_method='hist',
            random_state=42,
            n_jobs=-1
        )
        
        xgb_model.fit(train_feat_scaled, train_y, sample_weight=sample_weights)
        y_pred = xgb_model.predict(test_feat_scaled)
        
        acc = accuracy_score(test_y, y_pred)
        results['subject_names'].append(test_sub)
        results['accuracies'].append(acc)
        results['confusion_matrices'].append(confusion_matrix(test_y, y_pred, labels=[0, 1, 2]))
        results['feature_importances'].append(xgb_model.feature_importances_)
        
        all_y_true.extend(test_y)
        all_y_pred.extend(y_pred)
        print(f"Test Accuracy: {acc:.4f}")
        
    return all_y_true, all_y_pred, results

# ==========================================
# 4. Results Visualization & Evaluation Dashboard
# ==========================================
def plot_and_evaluate(y_true, y_pred, results):
    if results is None: return
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS (XGBoost + Calibration)")
    print("="*50)
    
    mean_acc = np.mean(results['accuracies'])
    print(f"Overall Mean Accuracy: {mean_acc:.4f} ± {np.std(results['accuracies']):.4f}\n")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'BCI Classifier (XGBoost + {Config.CALIBRATION_RATIO*100:.0f}% Calibration)', fontsize=16)
    
    # 1. Accuracy
    subject_names = results['subject_names']
    axes[0].bar(subject_names, results['accuracies'], 
                color=['green' if acc >= 0.7 else 'orange' if acc >= 0.65 else 'red' for acc in results['accuracies']])
    axes[0].set_title('Accuracy by Subject')
    axes[0].set_ylabel('Accuracy')
    axes[0].axhline(y=np.mean(results['accuracies']), color='r', linestyle='--', label=f'Mean: {np.mean(results["accuracies"]):.3f}')
    axes[0].axhline(y=0.65, color='b', linestyle='--', label=f'Target: 0.65')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis='x', rotation=45) 
    
    # 2. Confusion Matrix
    sns.heatmap(np.sum(results['confusion_matrices'], axis=0), annot=True, fmt='d', cmap='Blues',
                xticklabels=['Relax', 'Focus', 'Blink'], yticklabels=['Relax', 'Focus', 'Blink'], ax=axes[1])
    axes[1].set_title('Overall Confusion Matrix')
    
    # 3. Feature Importance
    avg_importances = np.mean(results['feature_importances'], axis=0)
    indices = np.argsort(avg_importances)[::-1]
    sns.barplot(x=avg_importances[indices], y=[FEATURE_NAMES[i] for i in indices], ax=axes[2], palette="magma")
    axes[2].set_title('XGBoost Feature Importance')
    
    plt.tight_layout()
    save_path = os.path.join(Config.RUN_DIR, "xgb_bci_dashboard.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Dashboard saved to '{save_path}'")
    plt.show()

# ==========================================
# 5. Train Final Model + Hyperparameter Search (HPO)
# ==========================================
def train_and_save_final_model():
    dataset_dict = load_all_data()
    if not dataset_dict: return
        
    print("\nPreparing to train the final deployment model (initiating automatic hyperparameter search HPO)...")
    
    feat_list, y_list = [], []
    for sub in dataset_dict:
        feat_list.append(dataset_dict[sub]['X_feat'])
        y_list.append(dataset_dict[sub]['y'])
        
    train_feat = np.vstack(feat_list)
    train_y = np.hstack(y_list)
    
    scaler = StandardScaler()
    train_feat_scaled = scaler.fit_transform(train_feat)
    joblib.dump(scaler, os.path.join(Config.RUN_DIR, "feature_scaler.pkl"))
    
    sample_weights = compute_sample_weight(class_weight='balanced', y=train_y)
    
    param_grid = {
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    base_xgb = xgb.XGBClassifier(objective='multi:softprob', num_class=3, tree_method='hist', random_state=42)
    
    print("Executing RandomizedSearchCV to find optimal hyperparameters ...")
    random_search = RandomizedSearchCV(
        estimator=base_xgb, 
        param_distributions=param_grid, 
        n_iter=10,        
        scoring='accuracy', 
        cv=3,             
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    fit_params = {"sample_weight": sample_weights}
    random_search.fit(train_feat_scaled, train_y, **fit_params)
    
    best_model = random_search.best_estimator_
    print(f"Best parameters found: {random_search.best_params_}")
    
    best_score = random_search.best_score_
    print(f"In the hyperparameter search process, the highest cross-validation accuracy: {best_score:.4f}")
    
    model_save_path = os.path.join(Config.RUN_DIR, "bci_xgb_model_final.pkl")
    joblib.dump(best_model, model_save_path)
    print(f"Final optimized model saved to: {model_save_path}")

# ==========================================
# Main Program
# ==========================================
if __name__ == "__main__":
    os.makedirs(Config.RUN_DIR, exist_ok=True)
    if execute_loso:
        y_true, y_pred, results = leave_one_subject_out_cv()
        if y_true is not None:
            plot_and_evaluate(y_true, y_pred, results)
            
    if train_final_model:        
        train_and_save_final_model()
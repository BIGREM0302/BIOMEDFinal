import serial
import time
import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew # 🌟 新增 skew
from collections import deque
import warnings
import threading
import joblib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

warnings.filterwarnings('ignore')

# ==========================================
# 1. 參數設定
# ==========================================
COM_PORT = 'COM4'  
BAUD_RATE = 57600
# 🌟 改成你的 XGBoost 模型路徑 (注意副檔名是 .pkl)
MODEL_PATH = "model/XGB_20260522_022835/bci_xgb_model_final.pkl" 
SCALER_PATH = "model/XGB_20260522_022835/feature_scaler.pkl"

# ... (中間參數保留不變) ...

# 🌟 刪除 DEVICE = torch.device(...) 這行
# 🌟 刪除整個 DualInputBCINet 與 SELayer 的類別定義 (我們用不到神經網路了)

SAMPLING_RATE = 512
WINDOW_SECONDS = 1.0       # 1秒推論視窗 (512 samples)
STEP_SECONDS = 0.25        # 每次推論的滑動步長 0.25秒
WINDOW_SAMPLES = int(SAMPLING_RATE * WINDOW_SECONDS)
STEP_SAMPLES = int(SAMPLING_RATE * STEP_SECONDS)

BLINK_AMP_THRESHOLD = 2000
UNUSUAL_THRESHOLD = 15000

# 畫圖專用設定
PLOT_SECONDS = 3.0         # 畫面總共顯示 3 秒的原始波形
PLOT_SAMPLES = int(SAMPLING_RATE * PLOT_SECONDS)
FEATURE_HISTORY_SECONDS = 10.0
FEATURE_HISTORY_LEN = int(FEATURE_HISTORY_SECONDS / STEP_SECONDS) 

CLASS_NAMES = ['Relax', 'Focus', 'Blink']
COLOR_MAP = {'Relax': '#2ecc71', 'Focus': '#3498db', 'Blink': '#e74c3c', 'Waiting': '#95a5a6'}

# ==========================================
# 全域控制開關 (用來通知背景執行緒何時該停下來)
# ==========================================
running_flag = True


# ==========================================
# 3. 預處理與特徵萃取
# ==========================================
def bandpass_filter(data, lowcut=0.5, highcut=100.0, fs=SAMPLING_RATE, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

# ==========================================
# 3. 預處理與特徵萃取
# ==========================================
def extract_single_segment_features(seg):
    # Hjorth Parameters
    activity = np.var(seg) + 1e-7
    diff1 = np.diff(seg)
    diff2 = np.diff(diff1)
    var_diff1 = np.var(diff1) + 1e-7
    var_diff2 = np.var(diff2) + 1e-7
    mobility = np.sqrt(var_diff1 / activity)
    complexity = np.sqrt(var_diff2 / var_diff1) / mobility
    
    # 頻域特徵
    nperseg = min(len(seg), int(SAMPLING_RATE * 2.0)) 
    freqs, psd = signal.welch(seg, fs=SAMPLING_RATE, nperseg=nperseg)
    
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

    # 時域特徵
    zcr = np.sum(np.diff(np.sign(seg)) != 0) / len(seg)
    rms = np.sqrt(np.mean(seg**2))
    
    # 🌟 嚴格對齊 XGBoost 訓練時的 18 維特徵順序
    current_feature = [
        mobility, complexity,            # 0, 1
        rel_theta, rel_beta,             # 2, 3
        engagement_index, theta_beta_ratio, # 4, 5
        alpha_peak_power,                # 6
        kurt, skew_val, p2p_norm,        # 7, 8, 9
        #np.log10(activity),              # 10
        # np.max(np.abs(seg)),             # 11
        spectral_entropy, zcr,           # 12, 13
        # rms,                             # 14
        alpha_beta_ratio, rel_delta, rel_gamma # 15, 16, 17
    ]
    return np.array(current_feature)

class RealTimePredictor:
    def __init__(self, model_path, scaler_path):
        print(f"載入 XGBoost 模型權重與 Scaler: {model_path} ...")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, raw_data):
        # 計算 Peak-to-Peak 振幅
        raw_centered = raw_data - np.mean(raw_data)
        p2p = np.max(raw_centered) - np.min(raw_centered)

        # 濾波與正規化
        filtered = bandpass_filter(raw_data)
        normalized = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
        
        # 1. 萃取特徵與縮放
        raw_feats = extract_single_segment_features(normalized)
        feat_array = raw_feats.reshape(1, -1)
        scaled_feat = self.scaler.transform(feat_array)

        # ==========================================
        # 🛡️ 防禦機制 1：異常大雜訊 (對齊 real_bci.py)
        # ==========================================
        if p2p > 15000:
            return 1, 0.0, raw_feats # 視為無效或低信心，照常回傳特徵以供畫圖

        # 2. XGBoost 預測
        probs = self.model.predict_proba(scaled_feat)[0]
        pred_class = np.argmax(probs)
        
        # ==========================================
        # 🛡️ 防禦機制 2：物理振幅門檻 (對齊 real_bci.py 的 1500)
        # ==========================================
        # 如果模型很有信心是眨眼，但波形起伏太小，我們剝奪它成為眨眼的資格
        if pred_class == 2 and p2p < 1500: 
            probs[2] = 0.0
            if np.sum(probs) > 0:
                probs = probs / np.sum(probs)
            pred_class = np.argmax(probs)
        
        # 最終定案
        confidence = probs[pred_class]
        
        return pred_class, confidence, raw_feats

# ==========================================
# 4. 全域變數與資料共享
# ==========================================
plot_buffer = deque(np.zeros(PLOT_SAMPLES), maxlen=PLOT_SAMPLES)
buffer_lock = threading.Lock()

current_state = {
    'label': 'Waiting',
    'confidence': 0.0,
    'color': COLOR_MAP['Waiting']
}

feature_history = {
    'complexity': deque(np.zeros(FEATURE_HISTORY_LEN), maxlen=FEATURE_HISTORY_LEN),
    'engagement': deque(np.zeros(FEATURE_HISTORY_LEN), maxlen=FEATURE_HISTORY_LEN),
    'rel_alpha': deque(np.zeros(FEATURE_HISTORY_LEN), maxlen=FEATURE_HISTORY_LEN)
}

# ==========================================
# 5. 背景執行緒：處理藍牙讀取與推論
# ==========================================
def serial_worker():

    global running_flag
    predictor = RealTimePredictor(MODEL_PATH, SCALER_PATH)
    new_samples_count = 0 

    ser = None
    
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        print(f"📡 成功連接至 {COM_PORT}，開始接收腦波訊號...")
        
        # 這裡放你原本的 while True 讀取迴圈
        while running_flag:
            try:
                byte = ser.read(1)
                if not byte:
                    continue
                if byte == b'\xaa':
                    if ser.read(1) == b'\xaa':
                        length = ord(ser.read(1))
                        if length < 170:
                            payload = ser.read(length)
                            checksum = ord(ser.read(1))
                            
                            i = 0
                            while i < len(payload):
                                code = payload[i]
                                if code == 0x80: 
                                    val_len = payload[i+1]
                                    raw_val = (payload[i+2] << 8) | payload[i+3]
                                    if raw_val > 32768:
                                        raw_val -= 65536
                                    
                                    with buffer_lock:
                                        plot_buffer.append(raw_val)
                                    new_samples_count += 1
                                    
                                    if new_samples_count >= STEP_SAMPLES:
                                        with buffer_lock:
                                            if len(plot_buffer) >= WINDOW_SAMPLES:
                                                window_data = np.array(list(plot_buffer)[-WINDOW_SAMPLES:])
                                            else:
                                                window_data = None
                                                
                                        if window_data is not None:
                                            # 執行深度學習預測，同時拿回 13 維特徵
                                            pred_idx, conf, feats = predictor.predict(window_data)
                                            label = CLASS_NAMES[pred_idx]
                                            
                                            current_state['label'] = label
                                            current_state['confidence'] = conf
                                            current_state['color'] = COLOR_MAP[label]
                                            
                                            # 🌟 更新對應 extract_single_segment_features 回傳的索引:
                                            # Index 1: complexity, Index 4: engagement_index, Index 3: rel_beta
                                            comp = feats[1]
                                            eng = feats[4]
                                            r_beta = feats[3] # 🌟 改抓 rel_beta
                                            
                                            with buffer_lock:
                                                feature_history['complexity'].append(comp)
                                                feature_history['engagement'].append(eng)
                                                feature_history['rel_alpha'].append(r_beta) # 變數名沿用，但裝的是 Beta
                                                
                                            print(f"[{label:5}] 信心: {conf:.2f} | 專注度: {eng:.2f} | 複雜度: {comp:.2f}")
                                            
                                        new_samples_count = 0 
                                    i += (val_len + 2)
                                elif code == 0x02:  
                                    i += 2
                                elif code in [0x04, 0x05]:
                                    i += 2
                                else:
                                    i += 1
            except Exception as e:
                print(f"Serial Worker 錯誤: {e}")
                break
            
    except serial.SerialException as e:
        print(f"❌ 無法開啟或中斷了 {COM_PORT}: {e}")
        
    except KeyboardInterrupt:
        # 當你按 Ctrl+C 時會觸發這裡
        print("\n🛑 收到強制停止指令...")
        
    finally:
        if ser is not None and getattr(ser, 'is_open', False):
            ser.close()
            print("🔌 藍牙通道已安全釋放，下次可直接重新執行！")

    

# ==========================================
# 6. 主程式：Matplotlib 即時繪圖
# ==========================================
def main():
    global running_flag
    worker_thread = threading.Thread(target=serial_worker, daemon=False)
    worker_thread.start()
    time.sleep(1)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    
    ax_main = axes[0]
    ax_comp = axes[1]
    ax_eng = axes[2]
    ax_alpha = axes[3]
    
    x_axis_main = np.linspace(-PLOT_SECONDS, 0, PLOT_SAMPLES)
    line_main, = ax_main.plot(x_axis_main, np.zeros(PLOT_SAMPLES), lw=1.2, color='black')
    ax_main.set_xlim(-PLOT_SECONDS, 0)
    ax_main.set_ylim(-500, 500)
    ax_main.set_title("Real-Time EEG & Feature Analysis (Dual-Input)", fontsize=12, fontweight='bold')
    ax_main.set_ylabel("Amplitude")
    ax_main.grid(True, linestyle='--', alpha=0.5)
    
    inference_window_span = ax_main.axvspan(-WINDOW_SECONDS, 0, color=COLOR_MAP['Waiting'], alpha=0.3)
    pred_text = ax_main.text(0.85, 0.90, "Waiting...", transform=ax_main.transAxes, 
                             ha='center', va='center', fontsize=12, fontweight='bold', 
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    x_axis_feat = np.linspace(-FEATURE_HISTORY_SECONDS, 0, FEATURE_HISTORY_LEN)
    
    line_comp, = ax_comp.plot(x_axis_feat, np.zeros(FEATURE_HISTORY_LEN), color='purple', lw=1.5)
    ax_comp.set_ylabel("Complexity")
    ax_comp.set_xlim(-FEATURE_HISTORY_SECONDS, 0)
    ax_comp.grid(True, linestyle='--', alpha=0.5)

    line_eng, = ax_eng.plot(x_axis_feat, np.zeros(FEATURE_HISTORY_LEN), color='blue', lw=1.5)
    ax_eng.set_ylabel("Engagement")
    ax_eng.set_xlim(-FEATURE_HISTORY_SECONDS, 0)
    ax_eng.grid(True, linestyle='--', alpha=0.5)

    line_alpha, = ax_alpha.plot(x_axis_feat, np.zeros(FEATURE_HISTORY_LEN), color='green', lw=1.5)
    ax_alpha.set_ylabel("Rel. Beta") # 🌟 改名為 Rel. Beta
    ax_alpha.set_xlim(-FEATURE_HISTORY_SECONDS, 0)
    ax_alpha.set_xlabel("Time (seconds)")
    ax_alpha.grid(True, linestyle='--', alpha=0.5)

    def update_plot(frame):
        with buffer_lock:
            y_data = np.array(plot_buffer)
            hist_comp = np.array(feature_history['complexity'])
            hist_eng = np.array(feature_history['engagement'])
            hist_alpha = np.array(feature_history['rel_alpha'])
            
        y_plot = y_data - np.mean(y_data)
        line_main.set_ydata(y_plot)
        y_max = np.max(np.abs(y_plot))
        if y_max > 50: 
            ax_main.set_ylim(-y_max * 1.5, y_max * 1.5)

        inference_window_span.set_color(current_state['color'])
        label = current_state['label']
        conf = current_state['confidence']
        if label == 'Waiting':
            pred_text.set_text("Waiting...")
        else:
            pred_text.set_text(f"{label}\n({conf:.2f})")
            pred_text.set_color(current_state['color'])

        line_comp.set_ydata(hist_comp)
        if np.max(hist_comp) > 0: ax_comp.set_ylim(np.min(hist_comp)*0.9, np.max(hist_comp)*1.1)

        line_eng.set_ydata(hist_eng)
        if np.max(hist_eng) > 0: ax_eng.set_ylim(np.min(hist_eng)*0.9, np.max(hist_eng)*1.1)

        line_alpha.set_ydata(hist_alpha)
        if np.max(hist_alpha) > 0: ax_alpha.set_ylim(np.min(hist_alpha)*0.9, np.max(hist_alpha)*1.1)

        return line_main, inference_window_span, pred_text, line_comp, line_eng, line_alpha

    # ==========================================
    # 🌟 這裡開始是修正後的順序：先綁定，再 Show
    # ==========================================
    
    # 2. 定義監聽器
    def on_close(event):
        global running_flag
        print("\n👋 偵測到圖表視窗關閉，正在觸發安全釋放機制...")
        running_flag = False 
        
    # 3. 註冊監聽器到畫布
    fig.canvas.mpl_connect('close_event', on_close)

    # 設定動畫與排版
    plt.tight_layout()
    ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=False)
    
    # 🌟 整個 main 函數裡只能有這「唯一」一個 plt.show()
    plt.show() 
    
    # ==========================================
    # 4. 當視窗被關掉、plt.show() 解除阻擋後，程式才會走到這裡
    # ==========================================
    print("⏳ 正在等待背景執行緒釋放藍牙... (最多等待 2 秒)")
    worker_thread.join(timeout=2)
    print("🏁 資源清理完畢，程式安全結束。")

if __name__ == "__main__":
    main()
"""
for XGboost model
"""

import serial
import time
import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew
from collections import deque
import threading
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

# ==========================================
# 1. 參數設定
# ==========================================
COM_PORT = 'COM3'  # ⚠️ 依據實際「連出 (Outgoing)」狀況修改
BAUD_RATE = 57600
# ⚠️ 改成你 XGBoost 訓練出來的 .pkl 檔案路徑
MODEL_PATH = "./model/XGB_20260522_022835/bci_xgb_model_final.pkl"
SCALER_PATH = "./model/XGB_20260522_022835/feature_scaler.pkl"  

print("目前工作目錄:", os.getcwd())
print("MODEL 是否存在:", os.path.exists(MODEL_PATH))
print("SCALER 是否存在:", os.path.exists(SCALER_PATH))

SAMPLING_RATE = 512
WINDOW_SECONDS = 1.0
STEP_SECONDS = 0.25
WINDOW_SAMPLES = int(SAMPLING_RATE * WINDOW_SECONDS)
STEP_SAMPLES = int(SAMPLING_RATE * STEP_SECONDS)

# ==========================================
# 2. 資料處理與 18 維特徵萃取 (對齊 XGBoost)
# ==========================================
def bandpass_filter(data, lowcut=0.5, highcut=100.0, fs=SAMPLING_RATE, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def extract_single_segment_features(seg):
    activity = np.var(seg) + 1e-7
    diff1 = np.diff(seg)
    diff2 = np.diff(diff1)
    var_diff1 = np.var(diff1) + 1e-7
    var_diff2 = np.var(diff2) + 1e-7
    mobility = np.sqrt(var_diff1 / activity)
    complexity = np.sqrt(var_diff2 / var_diff1) / mobility
    
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

    zcr = np.sum(np.diff(np.sign(seg)) != 0) / len(seg)
    rms = np.sqrt(np.mean(seg**2))
    
    current_feature = [
        mobility, complexity, 
        rel_theta, rel_beta, 
        engagement_index, theta_beta_ratio,
        alpha_peak_power,
        kurt, skew_val, p2p_norm,
        # np.log10(activity), 
        # np.max(np.abs(seg)),             # 11
        spectral_entropy, zcr, 
        # rms, 
        alpha_beta_ratio, rel_delta, rel_gamma
    ]
    return np.array(current_feature)

# ==========================================
# 3. XGBoost 即時預測器 (含防禦機制)
# ==========================================
class RealTimePredictor:
    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, raw_data):
        raw_centered = raw_data - np.mean(raw_data)
        p2p = np.max(raw_centered) - np.min(raw_centered)

        if p2p > 15000:
            return 1, 0.0 # 異常雜訊視為 Focus 或無效

        filtered = bandpass_filter(raw_data)
        normalized = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
        
        # 特徵萃取與縮放
        feat_array = extract_single_segment_features(normalized).reshape(1, -1)
        scaled_feat = self.scaler.transform(feat_array)
        
        # XGBoost 預測
        probs = self.model.predict_proba(scaled_feat)[0]
        pred_class = np.argmax(probs)
        """
        # 🛡️ 雙重防禦機制：避免假眨眼
        if pred_class == 2 and probs[2] < 0.70:
            probs[2] = 0.0
            if sum(probs) > 0: probs = probs / sum(probs)
            pred_class = np.argmax(probs) 
        """
            
        if pred_class == 2 and p2p < 1500:
            probs[2] = 0.0
            if sum(probs) > 0: probs = probs / sum(probs)
            pred_class = np.argmax(probs)
        

        confidence = probs[pred_class]
        return pred_class, confidence

# ==========================================
# 4. 背景 BCI 引擎服務
# ==========================================
class RealBCI:
    def __init__(self, enable_monitor=True):
        self.enable_monitor = enable_monitor
        self.current_signal = "無"
        self.lock = threading.Lock()
        self.running = True # 🌟 新增控制開關
        
        self.class_map = {0: "放鬆", 1: "專注", 2: "眨眼"}
        self.data_buffer = deque(np.zeros(WINDOW_SAMPLES), maxlen=WINDOW_SAMPLES)

        self.consecutive_blinks = 0
        
        self.last_blink_time = 0.0
        self.blink_cooldown = 1.0  
        
        self.worker_thread = threading.Thread(target=self._bci_worker, daemon=False)
        self.worker_thread.start()

    def stop(self):
        """安全停止背景引擎並釋放藍牙"""
        self.running = False
        self.worker_thread.join(timeout=2)
        print("🏁 BCI 引擎已安全關閉。")

    def clear_signal(self):
        with self.lock:
            self.current_signal = "無"

    def get_signal(self):
        with self.lock:
            sig = self.current_signal
            if sig == "眨眼":
                self.current_signal = "無"
            return sig

    def _bci_worker(self):
        try:
            predictor = RealTimePredictor(MODEL_PATH, SCALER_PATH)
        except Exception as e:
            print(f"⚠️ 無法載入模型，請確認路徑。({e})")
            return

        ser = None
        try:
            ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
            print(f"📡 成功連接 BCI 設備 ({COM_PORT})，背景引擎啟動中...")
            
            new_samples_count = 0 
            total_samples_read = 0  

            while self.running: # 🌟 受控迴圈

                # 👇 加入這段「防護機制」：如果緩衝區堆積超過 1 秒鐘的資料量，直接清空
                if ser.in_waiting > 2048:
                    # print("⚠️ 偵測到腦波延遲堆積，捨棄舊資料以對齊即時反應！") # 測試時可打開這行觀察
                    ser.reset_input_buffer()
                    self.data_buffer.extend(np.zeros(WINDOW_SAMPLES))
                    new_samples_count = 0
                    total_samples_read = 0
                    continue
                # 👆 防護機制結束

                byte = ser.read(1)
                if not byte:
                    continue
                    
                if byte == b'\xaa' and ser.read(1) == b'\xaa':
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
                                if raw_val > 32768: raw_val -= 65536
                                
                                self.data_buffer.append(raw_val)
                                new_samples_count += 1
                                total_samples_read += 1 
                                
                                if new_samples_count >= STEP_SAMPLES:
                                    if total_samples_read < WINDOW_SAMPLES:
                                        new_samples_count = 0
                                        continue

                                    window_data = np.array(list(self.data_buffer))
                                    pred_idx, conf = predictor.predict(window_data)
                                    
                                    current_t = time.time()

                                    
                                    if pred_idx == 2: 
                                        if current_t - self.last_blink_time > self.blink_cooldown:
                                            detected_signal = "眨眼"
                                            self.last_blink_time = current_t
                                        else:
                                            detected_signal = "無" 
                                    else:
                                        detected_signal = self.class_map.get(pred_idx, "無")
                                    
                                    """
                                    ###################
                                    # 🌟 全新連續判定邏輯
                                    if pred_idx == 2: 
                                        self.consecutive_blinks += 1 # 看到眨眼，計數器 +1
                                        
                                        # 只有連續 2 次以上，且過了冷卻時間，才核准放行
                                        if self.consecutive_blinks >= 2 and (current_t - self.last_blink_time > self.blink_cooldown):
                                            detected_signal = "眨眼"
                                            self.last_blink_time = current_t
                                            self.consecutive_blinks = 0 # 觸發後重置計數
                                        else:
                                            # 如果只有 1 次，先扣住不發送，視為「無」
                                            detected_signal = "無" 
                                    else:
                                        # 只要中間斷掉了（變成專注或放鬆），計數器直接歸零
                                        self.consecutive_blinks = 0 
                                        detected_signal = self.class_map.get(pred_idx, "無")
                                    ###################
                                    """
                                    
                                    with self.lock:
                                        if self.current_signal != "眨眼":
                                            self.current_signal = detected_signal
                                        
                                    if self.enable_monitor:
                                        print(f"[BCI Monitor] {detected_signal} (Confidence: {conf:.2f})")
                                        
                                    new_samples_count = 0 
                                i += (val_len + 2)
                            elif code in [0x02, 0x04, 0x05]:
                                i += 2
                            else:
                                i += 1
                                
        except serial.SerialException as e:
            print(f"❌ 無法開啟或中斷了 {COM_PORT}: {e}")
        except Exception as e:
            print(f"Serial Worker 發生異常: {e}")
        finally:
            # 🌟 藍牙安全下車保護
            if ser is not None and getattr(ser, 'is_open', False):
                ser.close()
                print("🔌 藍牙通道已安全釋放。")

# --- 測試區塊 ---
if __name__ == "__main__":
    bci = RealBCI(enable_monitor=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n收到中斷指令，準備關閉...")
        bci.stop()
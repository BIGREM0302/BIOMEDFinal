import serial
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from collections import deque
import threading
import warnings

warnings.filterwarnings('ignore')

# 參數設定
COM_PORT = 'COM3'  # 依據實際狀況修改
BAUD_RATE = 57600
MODEL_PATH = "bci_model_final.pth"
SAMPLING_RATE = 512
WINDOW_SECONDS = 1.0
STEP_SECONDS = 0.25
WINDOW_SAMPLES = int(SAMPLING_RATE * WINDOW_SECONDS)
STEP_SAMPLES = int(SAMPLING_RATE * STEP_SECONDS)
BLINK_AMP_THRESHOLD = 2000
UNUSUAL_THRESHOLD = 15000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 模型架構 ---
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

def bandpass_filter(data, lowcut=1.0, highcut=40.0, fs=SAMPLING_RATE, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

class RealTimePredictor:
    def __init__(self, model_path):
        self.model = LightweightBCINet().to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()

    def predict(self, raw_data):
        raw_centered = raw_data - np.mean(raw_data)
        p2p = np.max(raw_centered) - np.min(raw_centered)
        zcr = np.sum(np.diff(np.sign(raw_centered)) != 0)
        diff_mean = np.mean(np.abs(np.diff(raw_centered)))

        if p2p > UNUSUAL_THRESHOLD:
            return 1, 0.0 # 異常雜訊視為 Focus 或無效

        is_intentional_blink = (zcr > 50) and (diff_mean > 40)
        print(f"P2P: {p2p:.1f} | ZCR: {zcr} | Diff: {diff_mean:.1f}")
        if is_intentional_blink or p2p > BLINK_AMP_THRESHOLD:
            return 2, 1.0 # 眨眼

        filtered = bandpass_filter(raw_data)
        normalized = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
        tensor_data = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(tensor_data)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
            
        pred_class = np.argmax(probs)
        confidence = probs[pred_class]
        
        # 消除誤判
        if pred_class == 2 and not is_intentional_blink:
            probs[2] = 0.0
            if sum(probs) > 0: probs = probs / sum(probs)
            pred_class = np.argmax(probs) 
        if pred_class == 0 and probs[0] <= 0.95:
            pred_class = 1

        return pred_class, confidence


class RealBCI:
    def __init__(self, enable_monitor=True):
        self.enable_monitor = enable_monitor
        self.current_signal = "無"
        self.lock = threading.Lock()
        
        self.class_map = {0: "放鬆", 1: "專注", 2: "眨眼"}
        self.data_buffer = deque(np.zeros(WINDOW_SAMPLES), maxlen=WINDOW_SAMPLES)
        
        # --- 新增：眨眼防連發冷卻機制 ---
        self.last_blink_time = 0.0
        self.blink_cooldown = 2.0  # 設定 2 秒內只能觸發一次有效的眨眼
        
        # 啟動背景讀取與推論執行緒
        self.worker_thread = threading.Thread(target=self._bci_worker, daemon=True)
        self.worker_thread.start()

    # --- 新增：提供給主程式清空殘留訊號的方法 ---
    def clear_signal(self):
        with self.lock:
            self.current_signal = "無"

    def _bci_worker(self):
        try:
            predictor = RealTimePredictor(MODEL_PATH)
        except Exception as e:
            print(f"⚠️ 無法載入模型 {MODEL_PATH}，請確認路徑。({e})")
            return

        try:
            ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
            print(f"📡 成功連接 BCI 設備 ({COM_PORT})，開始背景接收訊號...")
        except serial.SerialException:
            print(f"❌ 無法開啟 {COM_PORT}，請確認藍牙連線。")
            return

        new_samples_count = 0 
        total_samples_read = 0  # --- 新增：紀錄總共讀了多少 Data ---

        while True:
            try:
                byte = ser.read(1)
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
                                total_samples_read += 1 # 累加總讀取數
                                
                                # 每達到 0.25 秒的資料量，進行一次推論
                                if new_samples_count >= STEP_SAMPLES:
                                    
                                    # --- 【修正 1】暖機期：確保 Buffer 完全被真實訊號填滿才開始推論 ---
                                    if total_samples_read < WINDOW_SAMPLES:
                                        new_samples_count = 0
                                        continue

                                    window_data = np.array(list(self.data_buffer))
                                    pred_idx, conf = predictor.predict(window_data)
                                    
                                    # --- 【修正 2】滑動視窗防連發機制 ---
                                    current_t = time.time()
                                    if pred_idx == 2: # 如果偵測到眨眼
                                        if current_t - self.last_blink_time > self.blink_cooldown:
                                            detected_signal = "眨眼"
                                            self.last_blink_time = current_t
                                        else:
                                            detected_signal = "無" # 還在冷卻期，直接忽略
                                    else:
                                        detected_signal = self.class_map.get(pred_idx, "無")
                                    
                                    with self.lock:
                                        # 如果目前已經有眨眼卡在 current_signal 等待被主程式消耗，就不要被 放鬆/專注 蓋掉
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
            except Exception as e:
                print(f"Serial Worker 錯誤: {e}")
                break
            
    def get_signal(self):
        """提供給遊戲主迴圈呼叫，並消耗掉單次眨眼事件"""
        with self.lock:
            sig = self.current_signal
            # 如果是眨眼，回傳後立刻重置為"無"，避免在 0.25 秒內被重複觸發兩次
            if sig == "眨眼":
                self.current_signal = "無"
            return sig
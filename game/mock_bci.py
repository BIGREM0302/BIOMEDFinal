import time
import random

class MockBCI:
    def __init__(self, blink_cooldown=3.0):
        """
        初始化模擬器
        :param blink_cooldown: 兩次眨眼之間最短的間隔秒數
        """
        self.blink_cooldown = blink_cooldown
        self.last_blink_time = 0.0
        
        # 定義四種狀態
        self.states = ["無", "放鬆", "專注", "眨眼"]

    def get_signal(self):
        """
        獲取當前的模擬腦波訊號
        """
        current_time = time.time()
        
        # 檢查是否還在眨眼的冷卻時間內
        is_blink_on_cooldown = (current_time - self.last_blink_time) < self.blink_cooldown

        if is_blink_on_cooldown:
            # 如果在冷卻中，不把「眨眼」放入候選名單
            choices = ["無", "放鬆", "專注"]
            # 設定機率權重 (例如: 60%無, 20%放鬆, 20%專注)
            weights = [0.8, 0.05, 0.15] 
        else:
            # 如果冷卻完畢，加入「眨眼」
            choices = ["無", "放鬆", "專注", "眨眼"]
            # 設定機率權重 (例如: 50%無, 15%放鬆, 15%專注, 20%眨眼)
            weights = [0.6, 0.05, 0.25, 0.1]

        # 根據機率隨機抽出一個狀態
        signal = random.choices(choices, weights=weights, k=1)[0]
        
        # 如果抽中了眨眼，更新最後一次眨眼的時間
        if signal == "眨眼":
            self.last_blink_time = current_time
            
        return signal

# 測試區塊：如果你直接執行這個檔案，它會每秒印出一次訊號
if __name__ == "__main__":
    # 設定眨眼至少要間隔 3 秒
    bci_simulator = MockBCI(blink_cooldown=3.0) 
    
    print("🧠 開始模擬腦波訊號... (按 Ctrl+C 結束)")
    try:
        while True:
            sig = bci_simulator.get_signal()
            print(f"[{time.strftime('%X')}] 偵測到訊號: {sig}")
            time.sleep(1) # 每秒產生一筆資料
    except KeyboardInterrupt:
        print("\n模擬結束！")
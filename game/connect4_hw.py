import os
from os import environ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame
import random
import copy
import time
import math
import numpy as np

# 假設這些是你的 ML/BCI 模組
from mock_bci import MockBCI
import serial
import threading

# --- 狀態機常數 ---
UI_START = 0
UI_RULES = 1
UI_GAME = 2
UI_OVER = 3

class BoardController:
    def __init__(self, port='COM5', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.connected = False
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.05)
            self.connected = True
            print(f"🔌 成功連接實體棋盤 ({self.port})")
        except serial.SerialException:
            print(f"⚠️ 無法連接實體棋盤 ({self.port})，將僅使用螢幕顯示。")

    # [修改] 增加 ui_state 參數
    def send_state(self, env, ui_state):
        if not self.connected:
            return
            
        board_str = ""
        for y in range(env.boardY):
            for x in range(env.boardX):
                board_str += str(int(env.state[y][x]))
                
        focus = int(getattr(env, 'focus_level', 0))
        indicator = env.indicator_col
        ai_hint = env.ai_suggest_col if env.ai_suggest_col is not None else -1
        red_win = round(getattr(env, 'red_win_rate', 0.5), 2)
        ylw_win = round(getattr(env, 'yellow_win_rate', 0.5), 2)
        
        # 封包首位加入 ui_state
        packet = f"<{ui_state},{board_str},{focus},{indicator},{ai_hint},{red_win},{ylw_win}>\n"
        try:
            self.ser.write(packet.encode('utf-8'))
        except Exception as e:
            print(f"傳輸錯誤: {e}")

    def close(self):
        if self.connected and self.ser:
            self.ser.close()

class Connect4():
    def __init__(self):
        self.boardX = 7
        self.boardY = 6
        self.colour = 1
        self.yellowWin = False
        self.redWin = False
        self.done = False
        self.counter = 0
        self.state = np.zeros((self.boardY, self.boardX))
        self.red_win_rate = 0.5     
        self.yellow_win_rate = 0.5  

    def step(self, action):
        self.counter += 1
        reward = 0
        info = {}

        if self.state[0][action] != 0:
            self.done = True
            return self.state, -1, self.done, info

        if self.colour == 1:
            self.placer(action, True) 
            self.winCheck()
            if self.yellowWin:
                reward = 1
                self.done = True
            elif 0 not in self.state:
                self.done = True
            else:
                self.colour = 2 
        else:
            self.placer(action, False) 
            self.winCheck()
            if self.redWin:
                reward = -1
                self.done = True
            elif 0 not in self.state:
                self.done = True
            else:
                self.colour = 1 

        return self.state, reward, self.done, info

    def winCheck(self):
        connectAmount = 4
        YellowConnect = False
        RedConnect = False 

        # 水平
        for i in range(0,self.boardY):
            for z in range (0,self.boardX-connectAmount+1):
                c1, c2 = 0, 0
                for d in range (0,connectAmount):
                    if self.state[i][z+d] == 1: c1 += 1 
                    if self.state[i][z+d] == 2: c2 += 1 
                if c1 == connectAmount: YellowConnect = True
                if c2 == connectAmount: RedConnect = True

        # 垂直
        if not YellowConnect and not RedConnect:  
            for i in range (0,self.boardY-connectAmount+1):
                for z in range (0,self.boardX):
                    c1, c2 = 0, 0
                    for d in range (0,connectAmount):
                        if self.state[i+d][z] == 1: c1 += 1
                        if self.state[i+d][z] == 2: c2 += 1
                    if c1 == connectAmount: YellowConnect = True
                    if c2 == connectAmount: RedConnect = True

        # 正對角線
        if not YellowConnect and not RedConnect: 
            for i in range (0,self.boardY-connectAmount+1):
                for z in range (self.boardX-1,self.boardX-connectAmount-1,-1):
                    c1, c2 = 0, 0
                    for d in range (0,connectAmount):
                        if self.state[i+d][z-d] == 1: c1 += 1
                        if self.state[i+d][z-d] == 2: c2 += 1
                    if c1 == connectAmount: YellowConnect = True
                    if c2 == connectAmount: RedConnect = True

        # 負對角線
        if not YellowConnect and not RedConnect: 
            for i in range (0,self.boardY-connectAmount+1):
                for z in range (0,self.boardX-connectAmount+1):
                    c1, c2 = 0, 0
                    for d in range (0,connectAmount):
                        if self.state[i+d][z+d] == 1: c1 += 1
                        if self.state[i+d][z+d] == 2: c2 += 1
                    if c1 == connectAmount: YellowConnect = True
                    if c2 == connectAmount: RedConnect = True

        if YellowConnect: self.yellowWin = True
        if RedConnect: self.redWin = True
        return None

    def render(self,mode='human'):
        S = 0.7  
        
        def text(surface, fontFace, size, x, y, text_str, colour):
            font = pygame.font.SysFont('Courier', int(size * S), bold=True)
            text_obj = font.render(text_str, 1, colour)
            surface.blit(text_obj, (int(x * S), int(y * S)))
            
        if not hasattr(self, 'screen'):
            pygame.init()
            self.width, self.height = int(1000 * S), int(850 * S)
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("CYBERPUNK BCI CONNECT-4")

        if not hasattr(self, 'indicator_col'):
            self.indicator_col = 0
        if not hasattr(self, 'red_win_rate'):
            self.red_win_rate = 0.5
            self.yellow_win_rate = 0.5
        if not hasattr(self, 'focus_level'):
            self.focus_level = 0      
            self.ai_suggest_col = None 
            
        self.bg_color = (10, 12, 20)      
        self.grid_color = (20, 25, 40)    
        self.neon_cyan = (0, 255, 255)    
        self.neon_pink = (255, 20, 147)   
        self.neon_yellow = (255, 255, 0)  
        self.neon_green = (57, 255, 20)   
        self.white = (255, 255, 255)
        self.dark_grey = (50, 50, 60)
            
        if not hasattr(self, 'player_img'):
            avatar_size = (int(60 * S), int(60 * S))
            try:
                raw_player = pygame.image.load('player_avatar.png').convert_alpha()
                raw_ai = pygame.image.load('ai_avatar.png').convert_alpha()
                self.player_img = pygame.transform.scale(raw_player, avatar_size)
                self.ai_img = pygame.transform.scale(raw_ai, avatar_size)
                self.use_images = True
            except FileNotFoundError:
                self.use_images = False
                self.player_color = self.neon_cyan
                self.ai_color = self.neon_pink

        self.screen.fill(self.bg_color)

        offset_x = int(150 * S)
        offset_y = int(100 * S)
        current_mode = getattr(self, 'game_mode', 2) 
        
        box_w = max(1, int(2*S))
        
        player_text = "SYS.AI_1" if current_mode == 1 else "USER"
        pygame.draw.rect(self.screen, self.neon_cyan, (int(20*S), int(20*S), int(100*S), int(120*S)), box_w)
        text(self.screen, 'Courier', 20, 30, 150, player_text, self.neon_cyan)
        
        if getattr(self, 'use_images', False):
            self.screen.blit(self.player_img, (int(40*S), int(50*S)))  
        else:
            pygame.draw.circle(self.screen, self.player_color, (int(70*S), int(80*S)), int(30*S), max(2, int(3*S)))
        
        ai_text = "SYS.AI_2" if current_mode == 1 else "NEURAL.AI"
        pygame.draw.rect(self.screen, self.neon_cyan, (int(880*S), int(20*S), int(100*S), int(120*S)), box_w)
        text(self.screen, 'Courier', 20, 880, 150, ai_text, self.neon_cyan)
        
        if getattr(self, 'use_images', False):
            self.screen.blit(self.ai_img, (int(900*S), int(50*S)))
        else:
            pygame.draw.circle(self.screen, self.ai_color, (int(930*S), int(80*S)), int(30*S), max(2, int(3*S)))
        
        if current_mode in [2, 3]:
            bar_max_h = 300
            focus_h = int((self.focus_level / 100) * bar_max_h)
            pygame.draw.rect(self.screen, self.neon_green, (int(910*S), int((500 - focus_h)*S), int(40*S), int(focus_h*S))) 
            pygame.draw.rect(self.screen, self.neon_cyan, (int(910*S), int(200*S), int(40*S), int(300*S)), box_w)           
            text(self.screen, 'Courier', 18, 885, 520, f'SYNC:{self.focus_level:.1f}%', self.neon_green)

        turn_text = "SYS.RED_TURN" if self.colour == 2 else "SYS.YLW_TURN"
        active_text_color = self.neon_pink if self.colour == 2 else self.neon_yellow
        text(self.screen, 'Courier', 36, 380, 20, turn_text, active_text_color)

        pygame.draw.rect(self.screen, self.grid_color, (offset_x, offset_y, int(700*S), int(600*S)))
        pygame.draw.rect(self.screen, self.neon_cyan, (offset_x, offset_y, int(700*S), int(600*S)), max(2, int(4*S)))

        def draw_cyber_piece(cx, cy, color):
            pygame.draw.circle(self.screen, color, (cx, cy), int(42 * S), max(1, int(4*S)))
            pygame.draw.circle(self.screen, self.white, (cx, cy), int(15 * S))
            pygame.draw.circle(self.screen, color, (cx, cy), int(20 * S), max(1, int(2*S)))

        for i in range (0,self.boardX):
            for z in range (0,self.boardY):
                cx = offset_x + int((50 + i * 100) * S)
                cy = offset_y + int((50 + z * 100) * S)
                if self.state[z][i] == 0:
                    pygame.draw.circle(self.screen, self.dark_grey, (cx, cy), int(42 * S), max(1, int(2*S)))
                    pygame.draw.circle(self.screen, (30,30,40), (cx, cy), int(5 * S)) 
                elif self.state[z][i] == 1:
                    draw_cyber_piece(cx, cy, self.neon_yellow)
                elif self.state[z][i] == 2:
                    draw_cyber_piece(cx, cy, self.neon_pink)

        active_color = self.neon_yellow if self.colour == 1 else self.neon_pink
        flash_on = (pygame.time.get_ticks() % 400) < 200 
        
        if self.ai_suggest_col is not None and flash_on and current_mode in [2, 3]:
            col_x = offset_x + int((self.ai_suggest_col * 100) * S)
            pygame.draw.rect(self.screen, self.neon_cyan, (col_x, offset_y, int(100*S), int(600*S)), max(3, int(4*S)))
            pygame.draw.rect(self.screen, self.white, (col_x+2, offset_y+2, int(100*S)-4, int(600*S)-4), 1)

        for i in range(self.boardX):
            light_x = offset_x + int((50 + i * 100) * S)
            light_y = offset_y - int(30 * S)
            if self.ai_suggest_col == i and flash_on and current_mode in [2, 3]:
                pygame.draw.circle(self.screen, self.neon_cyan, (light_x, light_y), int(26 * S), max(2, int(4*S)))
                pygame.draw.circle(self.screen, self.white, (light_x, light_y), int(12 * S))
            elif i == self.indicator_col and current_mode != 3:
                pygame.draw.circle(self.screen, active_color, (light_x, light_y), int(15 * S))
            else:
                pygame.draw.circle(self.screen, self.dark_grey, (light_x, light_y), int(10 * S))

        bar_x, bar_y, bar_width, bar_height = 200, 780, 600, 20 
        pygame.draw.rect(self.screen, self.dark_grey, (int(bar_x*S), int(bar_y*S), int(bar_width*S), int(bar_height*S)))
        
        red_bar_width = bar_width * self.red_win_rate
        pygame.draw.rect(self.screen, self.neon_pink, (int(bar_x*S), int(bar_y*S), int(red_bar_width*S), int(bar_height*S)))
        
        yellow_bar_x = bar_x + red_bar_width
        yellow_bar_width = bar_width - red_bar_width
        pygame.draw.rect(self.screen, self.neon_yellow, (int(yellow_bar_x*S), int(bar_y*S), int(yellow_bar_width*S), int(bar_height*S)))
        
        text(self.screen, 'Courier', 18, bar_x, bar_y - 25, f"RED_PROB: {self.red_win_rate*100:.1f}%", self.neon_pink)
        text(self.screen, 'Courier', 18, bar_x + bar_width - 180, bar_y - 25, f"YLW_PROB: {self.yellow_win_rate*100:.1f}%", self.neon_yellow)

        self.winCheck()
        if self.yellowWin:
            text(self.screen, 'Courier', 60, 320, 350, '[ YLW VICTORY ]', self.neon_yellow)
        elif self.redWin:
            text(self.screen, 'Courier', 60, 320, 350, '[ RED VICTORY ]', self.neon_pink)
        elif 0 not in self.state:
            text(self.screen, 'Courier', 60, 320, 350, '[ STALEMATE ]', self.neon_cyan)

    def placer(self,action,colour):
        if colour == True:
            for i in range (self.boardY-1,-1,-1):
                if self.state[i][action] == 0:
                    self.state[i][action] = 1
                    break
        if colour == False:
            for i in range (self.boardY-1,-1,-1):
                if self.state[i][action] == 0:
                    self.state[i][action] = 2
                    break

    def getState(self): return self.state
    def getColour(self): return self.colour

    def reset(self):
        self.state = np.zeros((self.boardY, self.boardX))
        self.colour = 1
        self.redWin = False
        self.yellowWin = False
        self.done = False
        self.counter = 0
        self.red_win_rate = 0.5
        self.yellow_win_rate = 0.5
        return self.state


# ==========================================
# UI 繪製函式
# ==========================================
def draw_text(surface, text, size, x, y, color, center=False, alpha=255):
    font = pygame.font.SysFont('Courier', size, bold=True)
    text_surface = font.render(text, True, color)
    text_surface.set_alpha(alpha)
    if center:
        rect = text_surface.get_rect(center=(x, y))
        surface.blit(text_surface, rect)
    else:
        surface.blit(text_surface, (x, y))

def draw_start_screen(screen, w, h):
    screen.fill((10, 12, 20))
    draw_text(screen, "NEURAL INTERFACE", 30, w//2, h//2 - 80, (0, 255, 255), center=True)
    draw_text(screen, "CONNECT - 4", 70, w//2, h//2, (255, 255, 0), center=True)
    
    # 呼吸閃爍效果
    alpha = int((math.sin(pygame.time.get_ticks() / 300.0) + 1) * 127.5)
    draw_text(screen, "[ BLINK or CLICK TO INITIATE ]", 25, w//2, h//2 + 100, (255, 20, 147), center=True, alpha=alpha)

def draw_rules_screen(screen, w, h):
    screen.fill((10, 12, 20))
    draw_text(screen, "SYSTEM PROTOCOLS", 40, w//2, h//2 - 150, (0, 255, 255), center=True)
    
    rules = [
        "> 1. EYE BLINK : Execute Drop Action",
        "> 2. FOCUS     : Charge Synchronization Matrix",
        "> 3. RELAX     : Decrease Synchronization",
        "> 4. 100% SYNC : Unlocks AI Guidance Laser"
    ]
    for i, r in enumerate(rules):
        draw_text(screen, r, 22, w//2 - 250, h//2 - 50 + i*40, (255, 255, 255))
        
    alpha = int((math.sin(pygame.time.get_ticks() / 300.0) + 1) * 127.5)
    draw_text(screen, "[ BLINK TO ACCEPT PROTOCOLS ]", 25, w//2, h//2 + 150, (0, 255, 0), center=True, alpha=alpha)

def draw_game_over_screen(screen, w, h, env, relax_progress):
    screen.fill((10, 12, 20))
    draw_text(screen, "MATCH TERMINATED", 50, w//2, h//2 - 150, (0, 255, 255), center=True)
    
    if env.yellowWin:
        draw_text(screen, "USER VICTORY", 60, w//2, h//2 - 50, (255, 255, 0), center=True)
    elif env.redWin:
        draw_text(screen, "AI VICTORY", 60, w//2, h//2 - 50, (255, 20, 147), center=True)
    else:
        draw_text(screen, "STALEMATE", 60, w//2, h//2 - 50, (200, 200, 200), center=True)
        
    draw_text(screen, f"TOTAL TURNS: {env.counter}", 25, w//2, h//2 + 20, (255, 255, 255), center=True)
    
    draw_text(screen, ">>> BLINK TO RESTART <<<", 30, w//2, h//2 + 100, (0, 255, 0), center=True)
    
    # 顯示長按 Relax 關機的進度條
    draw_text(screen, "HOLD RELAX TO SHUTDOWN", 20, w//2, h//2 + 160, (255, 0, 0), center=True)
    bar_w = 300
    pygame.draw.rect(screen, (50, 50, 50), (w//2 - bar_w//2, h//2 + 190, bar_w, 20))
    pygame.draw.rect(screen, (255, 0, 0), (w//2 - bar_w//2, h//2 + 190, bar_w * relax_progress, 20))


# ==========================================
# 遊戲主迴圈
# ==========================================
if __name__ == '__main__':
    # 使用與原本 connect4.py 相同的 AI
    from CNNPlayWithSearch import heuristics

    # ==========================================
    # 1. Terminal 啟動選單 (在 Render 畫面之前)
    # ==========================================
    print("=============================================")
    print("🧠 BCI Connect 4 腦波對弈系統 - 核心啟動")
    print("=============================================")
    print("[1] AI vs AI 觀戰模式")
    print("[2] 玩家 vs AI 模式 (腦波下棋 + 腦波集氣)")
    print("[3] 智慧輔助模式 (滑鼠下棋 + 背景無縫 AI 提示)")
    print("=============================================")
    mode_choice = input("👉 請輸入模式代碼 (1, 2 或 3): ")
    game_mode = int(mode_choice) if mode_choice.strip() in ['1', '2', '3'] else 2

    print("\n=============================================")
    print("選擇腦波輸入來源：")
    print("[1] 模擬器 (MockBCI - 測試用)")
    print("[2] 真實藍牙設備 (RealBCI - 關閉即時繪圖確保效能)")
    print("=============================================")
    bci_choice = input("👉 請輸入來源代碼 (1 或 2): ")

    print("\n=============================================")
    print("畫面渲染設定：")
    print("[1] 開啟遊戲畫面 (預設)")
    print("[2] 關閉畫面 (Headless 模式，釋放最高效能給腦波預測)")
    print("=============================================")
    render_choice = input("👉 請輸入渲染代碼 (1 或 2): ")
    ENABLE_RENDER = (render_choice.strip() != '2')

    # 初始化 BCI
    if '2' in bci_choice:
        try:
            from real_bci import RealBCI
            bci = RealBCI(enable_monitor=True)
        except ImportError:
            print("⚠️ 找不到 real_bci 模組，退回使用 MockBCI")
            bci = MockBCI(blink_cooldown=3.0)
    else:
        bci = MockBCI(blink_cooldown=3.0)

    # ==========================================
    # 2. 選擇完畢，開始初始化 Pygame 與 UI 視窗
    # ==========================================
    pygame.init()
    S = 0.7
    width, height = int(1000 * S), int(850 * S)
    
    if ENABLE_RENDER:
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("CYBERPUNK BCI CONNECT-4")
    else:
        screen = None
        print("\n🚀 [Headless 模式啟動] 畫面渲染已關閉。遊戲邏輯與硬體通訊正常運作中...")
        print("💡 提示：在此模式下，請看實體棋盤進行遊戲。要結束請在 Terminal 按下 [Ctrl + C]。")
    
    clock = pygame.time.Clock()

    env = Connect4() 
    env.reset()
    env.game_mode = game_mode  # 將剛剛選擇的模式寫入 env
    env.indicator_col = 0
    env.focus_level = 0
    env.ai_suggest_col = None

    arduino_board = BoardController(port='COM5')

    running = True
    current_ui_state = UI_START # 啟動於開始畫面

    hint_calculated = False       
    env.pre_calculated_col = None 
    env.is_calculating = False    
    env.hint_printed = False      

    current_time = pygame.time.get_ticks()
    last_frame_time = current_time
    last_tick_time = current_time       
    bci_last_poll_time = current_time   
    ai_last_move_time = current_time    
    
    INDICATOR_INTERVAL = 1500           
    BCI_POLL_INTERVAL = 100             # 【優化 1】降低至 100ms，增加 BCI 反應即時性
    AI_MOVE_DELAY = 2000     
    
    #relax_shutdown_counter = 0
    #RELAX_SHUTDOWN_MAX = 12 # 大約連續 3 秒 (12 * 0.25s) 的 Relax 就關機
    relax_shutdown_counter = 0
    RELAX_SHUTDOWN_MAX = 18  # 【修改 3】降低為 6 (約 1.5 秒)，讓 Relax 集氣更容易結束遊戲
    ignore_blink_until = 0  # 【修改 1】新增冷卻計時器，用來阻擋殘留的眨眼訊號
    env.prev_indicator_col = 0 # 【修改 2】記錄上一個指示燈位置，用於延遲補償
    
    GAME_OVER_HOLD_MS = 10000 # 勝負出現後，先停留在棋盤畫面 10 秒再進結算頁
    game_over_started_at = None

    def background_ai(state, colour):
        """【優化 2】後台 AI 搜索線程 - 不阻塞主循環"""
        col, _ = heuristics(state, colour, MAX_NODES=14000)
        env.pre_calculated_col = col
        env.is_calculating = False

    last_arduino_send_time = 0
    
    # 【優化 3】為 BCI 信號創建獨立的讀取隊列，降低鎖爭用
    bci_signal_queue = []
    bci_signal_lock = threading.Lock()

    # 【優化 5】建立專用的 BCI 監聽線程，完全獨立於主遊戲循環
    def bci_listener_thread():
        """專用 BCI 監聽線程 - 不受主循環 FPS 影響，持續獲取訊號"""
        global bci_signal_queue, running
        last_poll = pygame.time.get_ticks()
        while running:
            current_t = pygame.time.get_ticks()
            if current_t - last_poll > BCI_POLL_INTERVAL:
                try:
                    sig = bci.get_signal()
                    with bci_signal_lock:
                        # 新訊號會覆蓋舊訊號（眨眼除外，會在獲取時自動清除）
                        bci_signal_queue.append(sig)
                        if len(bci_signal_queue) > 5:  # 防止隊列無限增長
                            bci_signal_queue.pop(0)
                except Exception as e:
                    print(f"BCI 監聽線程錯誤: {e}")
                last_poll = current_t
            else:
                time.sleep(0.01)  # 短暫讓出 CPU，避免忙輪詢

    bci_listener = threading.Thread(target=bci_listener_thread, daemon=True)
    bci_listener.start()
    time.sleep(0.5)  # 給 BCI 監聽線程啟動的時間
    
    try:
        while running:
            #clock.tick(60) # 👇 加入這行，強制鎖定最高 60 FPS，不再霸佔資源
            if not ENABLE_RENDER:
                clock.tick(60)
            current_time = pygame.time.get_ticks()
            delta_time = current_time - last_frame_time 
            last_frame_time = current_time              
            
            # --- 【優化 4】取得 BCI 訊號 - 改為非阻塞性獲取 ---
            # 從獨立的 BCI 監聽線程隊列中獲取訊號，不會阻塞主循環
            bci_signal = "無"
            with bci_signal_lock:
                if bci_signal_queue:
                    bci_signal = bci_signal_queue.pop(0)  # 取出最新訊號

            # --- 處理事件與滑鼠/鍵盤 UI 邏輯 ---
            if ENABLE_RENDER:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                    mouse_clicked = (event.type == pygame.MOUSEBUTTONDOWN)
                    space_pressed = (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE)
                    
                    if current_ui_state == UI_START:
                        # Mode 2 完全交給腦波控制，不用滑鼠/鍵盤跳過起始頁
                        if game_mode != 2 and (mouse_clicked or space_pressed):
                            current_ui_state = UI_RULES
                    elif current_ui_state == UI_RULES:
                        # Mode 2 完全交給腦波控制，不用滑鼠/鍵盤進入遊戲
                        if game_mode != 2 and (mouse_clicked or space_pressed):
                            env.reset()
                            current_ui_state = UI_GAME
                    elif current_ui_state == UI_GAME:
                        # 與 connect4.py 對齊：只有模式 3 允許滑鼠點擊下棋
                        if mouse_clicked and env.colour == 1 and not env.done and game_mode == 3:
                            col_width = 70 * S
                            target_col = int((event.pos[0] - 135*S) // col_width)
                            if 0 <= target_col < env.boardX and env.state[0][target_col] == 0:
                                env.step(target_col)
                                env.ai_suggest_col = None
                                hint_calculated = False 
                                if env.focus_level >= 100: env.focus_level -= 100
                                ai_last_move_time = current_time
                        
                        # Mode 2 不保留空白鍵測試落子；玩家落子只由 BCI 眨眼控制。
                        # Mode 3 的玩家落子只由上面的滑鼠點擊邏輯控制。
                                
                    elif current_ui_state == UI_OVER:
                        # Mode 2 的結算頁也交給腦波眨眼重新開始；其他模式可用滑鼠/空白鍵。
                        if game_mode != 2 and (mouse_clicked or space_pressed):
                            env.reset()
                            current_ui_state = UI_GAME
                            game_over_started_at = None
            else:
                # 即使無畫面，Pygame 還是需要 pump 來保持時鐘正常運作
                pygame.event.pump()
            
            # --- BCI 跨 UI 狀態統一處理 ---
            # 只有模式 2 才用眨眼控制 UI / 下棋，避免 AI vs AI 或滑鼠模式被 MockBCI 的眨眼自動跳過起始頁。
            if game_mode == 2 and bci_signal == "眨眼":
                if current_time > ignore_blink_until: # 【修改 1】確保訊號不在冷卻期內
                    if current_ui_state == UI_START:
                        current_ui_state = UI_RULES
                        ignore_blink_until = current_time + 1500 # 進入規則頁，給 1.5 秒冷卻
                        if hasattr(bci, 'clear_signal'): bci.clear_signal() # 強制清空殘留訊號
                        
                    elif current_ui_state == UI_RULES:
                        env.reset()
                        current_ui_state = UI_GAME
                        ignore_blink_until = current_time + 2000 # 進入遊戲，給 2 秒冷卻讓玩家準備
                        if hasattr(bci, 'clear_signal'): bci.clear_signal()
                        
                    elif current_ui_state == UI_GAME and env.colour == 1 and not env.done:
                        
                        target_col = env.indicator_col
                        
                        # 【修改 2】延遲補償：如果指示燈剛移動不到 500 毫秒，判定為玩家其實想下在「前一個」位置
                        if current_time - last_tick_time < 500 and hasattr(env, 'prev_indicator_col'):
                            if env.state[0][env.prev_indicator_col] == 0:  
                                target_col = env.prev_indicator_col

                        if env.state[0][target_col] == 0:
                            env.step(target_col)
                            env.ai_suggest_col = None
                            hint_calculated = False 
                            if env.focus_level >= 100: env.focus_level -= 100 
                            ai_last_move_time = current_time 
                            ignore_blink_until = current_time + 1000 # 遊戲中下子後，維持 1 秒冷卻即可
                            if hasattr(bci, 'clear_signal'): bci.clear_signal()
                            
                    elif current_ui_state == UI_OVER:
                        env.reset()
                        current_ui_state = UI_GAME
                        game_over_started_at = None
                        ignore_blink_until = current_time + 2000 # 重新開始後，給 2 秒冷卻
                        if hasattr(bci, 'clear_signal'): bci.clear_signal()
            
            # 結算畫面的「長按 Relax 關機」邏輯
            if current_ui_state == UI_OVER:
                if bci_signal == "放鬆":
                    relax_shutdown_counter += 1
                    if relax_shutdown_counter >= RELAX_SHUTDOWN_MAX:
                        print("📉 偵測到深度放鬆，系統安全關機...")
                        running = False
                elif bci_signal == "專注" or bci_signal == "眨眼":
                    relax_shutdown_counter = 0

            # ==========================================
            # 遊戲進行中邏輯 (UI_GAME)
            # ==========================================
            if current_ui_state == UI_GAME and not env.done:
                
                # 判斷是否為 AI 回合 (模式 1 兩邊都是 AI，模式 2/3 只有對手是 AI)
                is_ai_turn = (game_mode == 1) or (game_mode in [2, 3] and env.colour == 2)

                # --- 玩家回合邏輯 (Mode 2 & 3) ---
                if not is_ai_turn:
                    # 指示燈移動 (僅限模式 2)
                    if game_mode == 2:
                        if current_time - last_tick_time > INDICATOR_INTERVAL:
                            env.prev_indicator_col = env.indicator_col  # 【修改 2】移動前，先記錄上一個位置
                            for _ in range(env.boardX):
                                env.indicator_col = (env.indicator_col + 1) % env.boardX
                                if env.state[0][env.indicator_col] == 0:
                                    break 
                            last_tick_time = current_time

                    # 充能邏輯
                    if env.ai_suggest_col is None: 
                        if game_mode == 2:
                            if bci_signal == "專注":
                                env.focus_level = min(100, env.focus_level + 4) 
                            elif bci_signal == "放鬆":
                                env.focus_level = max(0, env.focus_level - 8)
                        elif game_mode == 3:
                            # 模式 3 自動隨時間充能
                            env.focus_level = min(100, env.focus_level + (delta_time / 3000.0) * 100)
                    
                    # 【優化 6】AI 提示運算 - 立即啟動，不等待任何條件
                    # 只要玩家回合開始且沒有提示顯示，就立即在後台啟動 heuristic 搜索
                    # 這使得 BCI Monitor 可以完全獨立運行，不受 heuristic 的阻塞
                    if env.ai_suggest_col is None and not env.is_calculating and not hint_calculated:
                        env.is_calculating = True
                        hint_calculated = True
                        env.pre_calculated_col = None
                        t = threading.Thread(target=background_ai, args=(copy.deepcopy(env.getState()), 1), daemon=True)
                        t.start()

                    # 觸發大絕招雷射 (共用)
                    if env.ai_suggest_col is None and env.focus_level >= 100:
                        if not env.is_calculating and env.pre_calculated_col is not None:
                            env.ai_suggest_col = env.pre_calculated_col

                # --- AI 對手回合邏輯 (包含 Mode 1 觀戰) ---
                else:
                    if current_time - ai_last_move_time >= AI_MOVE_DELAY:
                        import math
                        current_ai_colour = env.colour 
                        ai_target, eval_score = heuristics(env.getState(), current_ai_colour, MAX_NODES=10000)
                        env.step(ai_target)
                        
                        safe_eval = max(-10, min(10, eval_score))
                        win_prob = 1 / (1 + math.exp(-safe_eval))
                        if current_ai_colour == 2: 
                            env.red_win_rate = win_prob
                            env.yellow_win_rate = 1.0 - win_prob
                        else:
                            env.yellow_win_rate = win_prob
                            env.red_win_rate = 1.0 - win_prob
                        
                        env.ai_suggest_col = None
                        hint_calculated = False 
                        env.indicator_col = 0 
                        
                        if hasattr(bci, 'clear_signal'): bci.clear_signal()
                        bci_last_poll_time = current_time + 1000 
                        last_tick_time = current_time

            # --- 遊戲結束觸發切換狀態 ---
            # 不要立刻跳結算頁；先停留在棋盤上，讓玩家看清楚 Victory / Stalemate 與最後一步。
            if current_ui_state == UI_GAME and env.done:
                if game_over_started_at is None:
                    game_over_started_at = current_time
                elif current_time - game_over_started_at >= GAME_OVER_HOLD_MS:
                    current_ui_state = UI_OVER
                    relax_shutdown_counter = 0
                    game_over_started_at = None

            # ==========================================
            # 畫面繪製與硬體通訊
            # ==========================================
            if ENABLE_RENDER:
                if current_ui_state == UI_START:
                    draw_start_screen(screen, width, height)
                elif current_ui_state == UI_RULES:
                    draw_rules_screen(screen, width, height)
                elif current_ui_state == UI_OVER:
                    progress = min(1.0, relax_shutdown_counter / RELAX_SHUTDOWN_MAX)
                    draw_game_over_screen(screen, width, height, env, progress)
                elif current_ui_state == UI_GAME:
                    env.screen = screen 
                    env.render()

                pygame.display.update()


            if current_time - last_arduino_send_time >= 33:
                # --- 新增：攔截遊戲結束狀態，送出特定代碼給硬體 ---
                hw_ui_state = current_ui_state
                if current_ui_state == UI_GAME and env.done:
                    if env.yellowWin:
                        hw_ui_state = 4 # 黃方(玩家)獲勝
                    elif env.redWin:
                        hw_ui_state = 5 # 紅方(AI)獲勝
                    else:
                        hw_ui_state = 6 # 平手

                arduino_board.send_state(env, hw_ui_state)
                last_arduino_send_time = current_time

            # ⚠️ 無論有沒有畫面，硬體通訊都必須執行！
            #arduino_board.send_state(env, current_ui_state)
    except KeyboardInterrupt:
        print("\n🛑 收到強制停止指令，準備安全關機...")
        
    finally:
        # 確保任何情況下退出，都會安全釋放所有資源
        pygame.quit()
        arduino_board.close() 
        if hasattr(bci, 'stop'):
            bci.stop()
        print("🏁 系統已完全關閉。")
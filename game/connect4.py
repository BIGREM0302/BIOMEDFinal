import os
from os import environ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from posixpath import defpath

import pygame
import random
import copy
import time
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np

from stable_baselines3 import PPO
import torch as th
import gym
from stable_baselines3.common.utils import set_random_seed
import treelib
from treelib import Node, Tree
from scipy import special as sp
from posixpath import defpath
import re
from numpy import take
import pygame
import random
import copy
import time
import collections
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
from stable_baselines3 import PPO
import torch as th
from stable_baselines3.common.utils import set_random_seed
from treelib import Node, Tree
from scipy import special as sp
import pandas as pd
import numpy as np
import dask.dataframe as dd
import datetime
from scipy.signal import convolve2d
from multiprocessing import Pool
# from CNNMonte import monte
from newMonte import monte
from mock_bci import MockBCI

class Connect4(Env):
    def __init__(self):
        self.boardX = 7
        self.boardY = 6
        self.colour = 1
        self.yellowWin = False
        self.redWin = False
        self.done = False
        self.counter = 0
        self.action_space = Discrete(self.boardX)
        self.observation_space = Box(0, 4, shape = (self.boardY,self.boardX), dtype = np.int32)
        self.state = np.zeros((self.boardY, self.boardX))
        self.modelChoice = "0"
        self.red_win_rate = 0.5     # 紅方勝率 (0.0 ~ 1.0)
        self.yellow_win_rate = 0.5  # 黃方勝率


    def step(self, action):
        self.counter += 1
        reward = 0
        info = {}

        # 檢查是否為非法步
        if self.state[0][action] != 0:
            self.done = True
            return self.state, -1, self.done, info

        # 根據現在是誰的回合來落子
        if self.colour == 1:
            self.placer(action, True)  # 黃色 (先手)
            self.winCheck()
            if self.yellowWin:
                reward = 1
                self.done = True
            elif 0 not in self.state:
                self.done = True
            else:
                self.colour = 2  # 換對手
        else:
            self.placer(action, False) # 紅色 (後手)
            self.winCheck()
            if self.redWin:
                reward = -1
                self.done = True
            elif 0 not in self.state:
                self.done = True
            else:
                self.colour = 1  # 換對手

        self.render()
        return self.state, reward, self.done, info


    def winCheck(self):
        #Longways, breaks increase efficiency.
        connectAmount = 4
        YellowConnect = False
        RedConnect = False 

        for i in range(0,self.boardY):
            for z in range (0,self.boardX-connectAmount+1):
                counter1 = 0
                counter2 = 0
                for d in range (0,connectAmount):
                    if self.state[i][z+d] == 1:
                        counter1 += 1 
                    if self.state[i][z+d] == 2:
                        counter2 += 1 
                if counter1 == connectAmount:
                    YellowConnect = True
                    break
                if counter2 == connectAmount:
                    RedConnect = True
                    break
            if YellowConnect or RedConnect:
                break

        #Heightways

        #Throw in a checker before not to waste time.
        if not YellowConnect and not RedConnect:  
            for i in range (0,self.boardY-connectAmount+1):
                for z in range (0,self.boardX):
                    counter1 = 0
                    counter2 = 0
                    for d in range (0,connectAmount):
                        if self.state[i+d][z] == 1:
                            counter1 += 1
                        if self.state[i+d][z] == 2:
                            counter2 += 1
                    if counter1 == connectAmount:
                        YellowConnect = True
                        break
                    if counter2 == connectAmount:
                        RedConnect = True
                        break
                if YellowConnect or RedConnect:
                    break

        #Diagonal positive (not sure if this works for other lengths, probably should)

        if not YellowConnect and not RedConnect: 
            for i in range (0,self.boardY-connectAmount+1):
                for z in range (self.boardX-1,self.boardX-connectAmount-1,-1):
                    counter1 = 0
                    counter2 = 0
                    for d in range (0,connectAmount):
                        if self.state[i+d][z-d] == 1:
                            counter1 += 1
                        if self.state[i+d][z-d] == 2:
                            counter2 += 1
                    if counter1 == connectAmount:
                        YellowConnect = True
                        break
                    if counter2 == connectAmount:
                        RedConnect = True
                        break
                if YellowConnect or RedConnect:
                    break

        #Diagonal negative

        if not YellowConnect and not RedConnect: 
            for i in range (0,self.boardY-connectAmount+1):
                for z in range (0,self.boardX-connectAmount+1):
                    counter1 = 0
                    counter2 = 0
                    for d in range (0,connectAmount):
                        if self.state[i+d][z+d] == 1:
                            counter1 += 1
                        if self.state[i+d][z+d] == 2:
                            counter2 += 1
                    if counter1 == connectAmount:
                        YellowConnect = True
                        break
                    if counter2 == connectAmount:
                        RedConnect = True
                        break
                if YellowConnect or RedConnect:
                    break

        if YellowConnect:
            self.yellowWin = True
        if RedConnect:
            self.redWin = True
        return None

    def legalCheck(self):
        if self.state is not None:
            possibleValues = []
            for i in range(0,self.boardX):
                if self.state[0][i] == 0:
                    possibleValues.append(i)
            return possibleValues
        else:
            return []

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
            # 設定頭像的長寬 (原本半徑是 30*S，所以直徑大約是 60*S)
            avatar_size = (int(60 * S), int(60 * S))
            
            try:
                # 載入圖片並保留透明度 (convert_alpha)
                # 請確保這兩張圖檔放在跟 connect4.py 同一個資料夾
                raw_player = pygame.image.load('player_avatar.png').convert_alpha()
                raw_ai = pygame.image.load('ai_avatar.png').convert_alpha()
                
                # 縮放到我們設定的大小
                self.player_img = pygame.transform.scale(raw_player, avatar_size)
                self.ai_img = pygame.transform.scale(raw_ai, avatar_size)
                self.use_images = True
            except FileNotFoundError:
                print("⚠️ 找不到圖片檔案，退回使用幾何圖形！")
                self.use_images = False
                self.player_color = self.neon_cyan
                self.ai_color = self.neon_pink

        

        self.screen.fill(self.bg_color)

        offset_x = int(150 * S)
        offset_y = int(100 * S)
        current_mode = getattr(self, 'game_mode', 2) 
        
        def draw_shape(shape, color, cx, cy, size):
            w = max(2, int(3*S)) 
            if shape == 'circle':
                pygame.draw.circle(self.screen, color, (cx, cy), size, w)
            elif shape == 'square':
                pygame.draw.rect(self.screen, color, (cx - size, cy - size, size*2, size*2), w)
            elif shape == 'triangle':
                pygame.draw.polygon(self.screen, color, [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)], w)
            elif shape == 'diamond':
                pygame.draw.polygon(self.screen, color, [(cx, cy - size), (cx - size, cy), (cx, cy + size), (cx + size, cy)], w)
            elif shape == 'heart':
                r = size // 2 + 2
                pygame.draw.circle(self.screen, color, (cx - size//2, cy - size//4), r, w)
                pygame.draw.circle(self.screen, color, (cx + size//2, cy - size//4), r, w)
                pygame.draw.polygon(self.screen, color, [(cx - size, cy - size//4), (cx + size, cy - size//4), (cx, cy + size)], w)

        box_w = max(1, int(2*S))
        
        player_text = "SYS.AI_1" if current_mode == 1 else "USER"
        pygame.draw.rect(self.screen, self.neon_cyan, (int(20*S), int(20*S), int(100*S), int(120*S)), box_w)
        text(self.screen, 'Courier', 20, 30, 150, player_text, self.neon_cyan)
        
        # 畫玩家頭像 (有圖片貼圖片，沒圖片畫預設圓形)
        if getattr(self, 'use_images', False):
            self.screen.blit(self.player_img, (int(40*S), int(50*S)))  
        else:
            pygame.draw.circle(self.screen, self.player_color, (int(70*S), int(80*S)), int(30*S), max(2, int(3*S)))
        
        ai_text = "SYS.AI_2" if current_mode == 1 else "NEURAL.AI"
        pygame.draw.rect(self.screen, self.neon_cyan, (int(880*S), int(20*S), int(100*S), int(120*S)), box_w)
        text(self.screen, 'Courier', 20, 880, 150, ai_text, self.neon_cyan)
        
        # 畫 AI 頭像 (有圖片貼圖片，沒圖片畫預設圓形)
        if getattr(self, 'use_images', False):
            self.screen.blit(self.ai_img, (int(900*S), int(50*S)))
        else:
            pygame.draw.circle(self.screen, self.ai_color, (int(930*S), int(80*S)), int(30*S), max(2, int(3*S)))
        
        # 【修改】模式 2 和 3 都需要顯示 Focus 電池條
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
        
        # 【修改】模式 2 和 3 都支援雷射鎖定框
        if self.ai_suggest_col is not None and flash_on and current_mode in [2, 3]:
            col_x = offset_x + int((self.ai_suggest_col * 100) * S)
            pygame.draw.rect(self.screen, self.neon_cyan, (col_x, offset_y, int(100*S), int(600*S)), max(3, int(4*S)))
            pygame.draw.rect(self.screen, self.white, (col_x+2, offset_y+2, int(100*S)-4, int(600*S)-4), 1)

        for i in range(self.boardX):
            light_x = offset_x + int((50 + i * 100) * S)
            light_y = offset_y - int(30 * S)
            # 優先畫出大絕招的雷射發射孔
            if self.ai_suggest_col == i and flash_on and current_mode in [2, 3]:
                pygame.draw.circle(self.screen, self.neon_cyan, (light_x, light_y), int(26 * S), max(2, int(4*S)))
                pygame.draw.circle(self.screen, self.white, (light_x, light_y), int(12 * S))
            # 【修改】只有在「非模式 3」的時候，才畫出會跑動的指示燈！
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

        pygame.display.update()

    def placer(self,action,colour):
        #Yellow
        if colour == True:
            for i in range (self.boardY-1,-1,-1):
                if self.state[i][action] == 0:
                    self.state[i][action] = 1
                    break
        #Red
        if colour == False:
            for i in range (self.boardY-1,-1,-1):
                if self.state[i][action] == 0:
                    self.state[i][action] = 2
                    break

    def getState(self):
        return self.state
    def getColour(self):
        return self.colour

    def reset(self):
        self.state = np.zeros((self.boardY, self.boardX))
        self.colour = 1
        self.reward = 0
        self.redWin = False
        self.yellowWin = False
        self.done = False
        self.modelChoice = str(random.randint(7,14))
        return self.state
    
bci = MockBCI(blink_cooldown=3.0)
if __name__ == '__main__':
    from game.CNNPlayWithSearch_old import heuristics
    import pygame
    import threading
    import copy
    from mock_bci import MockBCI  
    from real_bci import RealBCI 
    
    print("=============================================")
    print("🧠 BCI Connect 4 腦波對弈系統 - 核心啟動")
    print("=============================================")
    print("[1] AI vs AI 觀戰模式")
    print("[2] 玩家 vs AI 模式 (腦波下棋 + 腦波集氣)")
    print("[3] 智慧輔助模式 (滑鼠下棋 + 背景無縫 AI 提示)")
    print("=============================================")
    mode_choice = input("👉 請輸入模式代碼 (1, 2 或 3): ")
    game_mode = int(mode_choice) if mode_choice.strip() in ['1', '2', '3'] else 1

    print("\n=============================================")
    print("選擇腦波輸入來源：")
    print("[1] 模擬器 (MockBCI - 測試用)")
    print("[2] 真實藍牙設備 (RealBCI - 關閉即時繪圖確保效能)")
    print("=============================================")
    bci_choice = input("👉 請輸入來源代碼 (1 或 2): ")

    # 1. 系統初始化
    if '2' in bci_choice:
        bci = RealBCI(enable_monitor=True) 
    else:
        bci = MockBCI(blink_cooldown=3.0)

    env = Connect4() 
    env.reset()
    env.game_mode = game_mode 
    running = True
    
    # --- 狀態控制變數 ---
    hint_calculated = False       # AI 是否算完提示
    env.pre_calculated_col = None # AI 算出的最佳落子
    env.is_calculating = False    # AI 運算 Thread 鎖
    env.hint_printed = False      
    env.indicator_col = 0         # 選擇燈號位置

    # --- 時間控制變數 ---
    current_time = pygame.time.get_ticks()
    last_tick_time = current_time       # 控制燈號移動
    bci_last_poll_time = current_time   # 控制 BCI 讀取
    ai_last_move_time = current_time    # 控制 AI 對手下棋延遲
    turn_start_time = current_time      # 回合開始時間
    
    # --- 頻率設定 ---
    INDICATOR_INTERVAL = 1500           # 燈號每 1 秒移動一格
    BCI_POLL_INTERVAL = 250             # [修正] 對齊模型推論的 0.25 秒
    AI_MOVE_DELAY = 2000                 # 對手 AI 思考假動作延遲

    # 定義背景 AI 運算任務
    def background_ai(state, colour):
        col, _ = heuristics(state, colour)
        env.pre_calculated_col = col
        env.is_calculating = False

    env.render() 
    
    # 2. 主迴圈 (Game Loop)
    last_frame_time = pygame.time.get_ticks() # 新增這行
    while running:
        current_time = pygame.time.get_ticks()
        delta_time = current_time - last_frame_time # 新增這行
        last_frame_time = current_time              # 新增這行
        
        # ==========================================
        # [A] 畫面更新與事件處理
        # ==========================================
        env.render()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # (開發測試用) 鍵盤空白鍵強制眨眼、滑鼠強制下棋
            if event.type == pygame.KEYDOWN and game_mode == 2:
                if event.key == pygame.K_SPACE and env.colour == 1 and not env.done:
                    if env.state[0][env.indicator_col] == 0:
                        env.step(env.indicator_col)
                        env.ai_suggest_col = None
                        hint_calculated = False 
                        if env.focus_level >= 100: env.focus_level -= 100
            
            if event.type == pygame.MOUSEBUTTONDOWN and game_mode == 3:
                if env.colour == 1 and not env.done: 
                    # 簡化版的滑鼠點擊邏輯 (請依你的實際畫面比例微調)
                    col_width = 70 
                    target_col = int((event.pos[0] - 135) // col_width)
                    if 0 <= target_col < env.boardX and env.state[0][target_col] == 0:
                        env.step(target_col)
                        env.ai_suggest_col = None
                        hint_calculated = False 
                        if env.focus_level >= 100: env.focus_level -= 100

        if env.done:
            continue # 遊戲結束就停止更新邏輯，只維持畫面繪製

        # ==========================================
        # [B] 指示燈移動 (模式 1 & 2 使用)
        # ==========================================
        if game_mode != 3 and env.colour == 1:
            if current_time - last_tick_time > INDICATOR_INTERVAL:
                
                # 🌟 智慧尋找下一個可用的欄位 (最多找 7 次，避免無窮迴圈)
                for _ in range(env.boardX):
                    env.indicator_col = (env.indicator_col + 1) % env.boardX
                    # 檢查這一個欄位的最頂層 (index 0) 是否為 0 (代表還沒滿)
                    if env.state[0][env.indicator_col] == 0:
                        break # 找到了！跳出迴圈
                        
                last_tick_time = current_time

        """
        # ==========================================
        # [C] 真實 BCI 訊號處理 (僅限模式 2 且為玩家回合)
        # ==========================================
        if game_mode == 2 and env.colour == 1:
            if current_time - bci_last_poll_time > BCI_POLL_INTERVAL:
                signal = bci.get_signal()
                
                # 1. 動作指令：眨眼 -> 下棋
                if signal == "眨眼":
                    target_col = env.indicator_col
                    if env.state[0][target_col] == 0:
                        print(f"👁️ BCI 觸發(眨眼)！落子於第 {target_col} 排")
                        env.step(target_col)
                        
                        env.ai_suggest_col = None
                        hint_calculated = False 
                        # 🌟 眨眼下棋後，能量改為累積制扣除
                        if env.focus_level >= 100: env.focus_level -= 100 
                        ai_last_move_time = current_time 
                
                # 2. 數值指令：專注/放鬆 -> 充能
                elif env.ai_suggest_col is None: 
                    if signal == "專注":
                        # 🌟 調整速度：把原本的 15 改成 4 (越小充越慢，現在大約需要連續專注 6 秒才能滿)
                        env.focus_level = min(100, env.focus_level + 4) 
                    elif signal == "放鬆":
                        # 🌟 調整速度：放鬆時掉電的速度也同步調慢
                        env.focus_level = max(0, env.focus_level - 4)   
                
                bci_last_poll_time = current_time
        """
        # ==========================================
        # [C] 真實 BCI 訊號處理 (僅限模式 2 且為玩家回合)
        # ==========================================
        if game_mode == 2 and env.colour == 1:
            if current_time - bci_last_poll_time > BCI_POLL_INTERVAL:
                signal = bci.get_signal()
                
                # 1. 動作指令：眨眼 -> 下棋
                if signal == "眨眼":
                    target_col = env.indicator_col
                    if env.state[0][target_col] == 0:
                        print(f"👁️ BCI 觸發(眨眼)！落子於第 {target_col} 排")
                        env.step(target_col)
                        
                        env.ai_suggest_col = None
                        hint_calculated = False 
                        if env.focus_level >= 100: env.focus_level -= 100 
                        ai_last_move_time = current_time 
                
                # 2. 數值指令：專注/放鬆 -> 充能 (🌟 拿掉 ai_suggest_col 的鎖)
                else: 
                    if signal == "專注":
                        env.focus_level = min(100, env.focus_level + 4) 
                    elif signal == "放鬆":
                        env.focus_level = max(0, env.focus_level - 8)
                        
                        # 🌟 測試 Relax 專用機制：如果掉電低於 100%，就收回大絕招
                        if env.focus_level < 100 and env.ai_suggest_col is not None:
                            env.ai_suggest_col = None
                            env.hint_printed = False # 讓它下次滿電時可以再次印出提示
                            print("📉 [系統] 偵測到放鬆狀態，能量下降，雷射提示已暫時收回！")
                
                bci_last_poll_time = current_time

        # ==========================================
        # [D] 神經網路提示系統 (充能達標後給予提示)
        # ==========================================
        if game_mode in [2, 3] and env.colour == 1:
            
            # 玩家回合一開始，丟背景去算最佳解
            if not hint_calculated and not env.is_calculating:
                env.is_calculating = True
                hint_calculated = True
                env.pre_calculated_col = None
                env.hint_printed = False
                
                t = threading.Thread(target=background_ai, args=(copy.deepcopy(env.getState()), 1), daemon=True)
                t.start()
                turn_start_time = current_time


            # 模式 3 自動充能 (累積制)
            if game_mode == 3 and env.ai_suggest_col is None:
                # 🌟 調整速度：8000 代表需要 8000 毫秒 (8 秒) 才能充到 100%
                # 數字越大充越慢，你可以自由修改
                env.focus_level = min(100, env.focus_level + (delta_time / 3000.0) * 100)

            # 當充能達到 100% 且背景算完了，就發射雷射提示
            if env.ai_suggest_col is None and env.focus_level >= 100:
                if not env.is_calculating and env.pre_calculated_col is not None:
                    env.ai_suggest_col = env.pre_calculated_col
                    if not env.hint_printed:
                        print("🎯 系統同步完成，雷射鎖定最佳路徑！")
                        env.hint_printed = True

        # ==========================================
        # [E] 對手 AI 回合
        # ==========================================
        is_ai_turn = (game_mode == 1) or (game_mode in [2, 3] and env.colour == 2)
        
        if is_ai_turn:
            if current_time - ai_last_move_time >= AI_MOVE_DELAY:
                import math
                current_ai_colour = env.colour 
                ai_target, eval_score = heuristics(env.getState(), current_ai_colour)
                env.step(ai_target)
                
                # 計算勝率
                safe_eval = max(-10, min(10, eval_score))
                win_prob = 1 / (1 + math.exp(-safe_eval))
                if current_ai_colour == 2: 
                    env.red_win_rate = win_prob
                    env.yellow_win_rate = 1.0 - win_prob
                else:
                    env.yellow_win_rate = win_prob
                    env.red_win_rate = 1.0 - win_prob
                
                # 換玩家回合，重置所有準備變數
                env.ai_suggest_col = None
                hint_calculated = False 
                env.hint_printed = False
                env.indicator_col = 0 # 指示燈歸零
                
                # --- 【修正 4】強制清空 BCI 殘留訊號，並延遲玩家 BCI 生效時間 ---
                if hasattr(bci, 'clear_signal'):
                    bci.clear_signal()
                
                # 額外給予 800 毫秒的冷卻時間，確保玩家看清楚換回合了再開始收訊號
                bci_last_poll_time = current_time + 1000 
                last_tick_time = current_time

    pygame.quit()

    # 🌟 呼叫 RealBCI 的安全下車機制
    if hasattr(bci, 'stop'):
        bci.stop()
        
    print("🏁 遊戲視窗已關閉，藍牙通道已安全釋放！")
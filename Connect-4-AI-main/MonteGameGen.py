from sqlite3 import connect
from numpy.core.fromnumeric import take
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
import gym
from stable_baselines3.common.utils import set_random_seed
import treelib
from treelib import Node, Tree
import csv
import pandas as pd
from newMonte import monte
from multiprocessing import Pool


class Connect(Env):
    def __init__(self):
        self.boardX = 7
        self.boardY = 6
        self.colour = 1
        self.yellowWin = False
        self.redWin = False
        self.draw = False
        self.turn = True
        self.done = False
        self.counter = 0
        self.action_space = Discrete(self.boardX)
        self.observation_space = Box(0, 4, shape = (self.boardY,self.boardX), dtype = np.int32)
        self.state = np.zeros((self.boardY, self.boardX))
        self.modelChoice = "0"


    def step(self, action):
        reward = 5
        info = {}
        # self.render()

        self.winCheck()
        if self.yellowWin:
            reward = 1
            self.done = True
            return self.state,reward,self.done,info
        elif self.redWin:
            reward = -1
            self.done = True
            return self.state,reward,self.done,info
        elif 0 not in self.state:
            reward = 0
            self.done = True
            return self.state,reward,self.done,info
            

        # self.render()
        # time.sleep(1)

        if self.turn == True: 
            if  self.winTaker(self.possibilities(True)) == 9:
                self.placer(monte(self.state,1000),True)
                self.turn = not self.turn
                self.winCheck()
                if self.yellowWin:
                    reward = 1
                    self.done = True
                    return self.state,reward,self.done,info
                elif self.redWin:
                    reward = -1
                    self.done = True
                    return self.state,reward,self.done,info
                elif 0 not in self.state:
                    reward = 0
                    self.done = True
                    return self.state,reward,self.done,info
            else:
                self.placer(self.winTaker(self.possibilities(True)),True)
                self.turn = not self.turn
                self.winCheck()
                if self.yellowWin:
                    reward = 1
                    self.done = True
                    return self.state,reward,self.done,info
                elif self.redWin:
                    reward = -1
                    self.done = True
                    return self.state,reward,self.done,info
                elif 0 not in self.state:
                    reward = 0
                    self.done = True
                    return self.state,reward,self.done,info
                  
            return self.state,reward,self.done,info

        elif self.turn == False:
            #false makes it reds turn
            if self.winTaker(self.possibilities(False)) == 9:
                self.placer(monte(self.state,1000),False)
                self.winCheck()
                self.turn = not self.turn

                if self.yellowWin:
                    reward = 1
                    self.done = True
                    return self.state,reward,self.done,info
                elif self.redWin:
                    reward = -1
                    self.done = True
                    return self.state,reward,self.done,info
                elif 0 not in self.state:
                    reward = 0
                    self.done = True
                    return self.state,reward,self.done,info
            else:
                self.placer(self.winTaker(self.possibilities(False)),False)
                self.winCheck()
                self.turn = not self.turn
                if self.yellowWin:
                    reward = 1
                    self.done = True
                    return self.state,reward,self.done,info
                elif self.redWin:
                    reward = -1
                    self.done = True
                    return self.state,reward,self.done,info
                elif 0 not in self.state:
                    reward = 0
                    self.done = True
                    return self.state,reward,self.done,info

            return self.state,reward,self.done,info


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
        possibleValues = []
        for i in range(0,self.boardX):
            if self.state[0][i] == 0:
                possibleValues.append(i)
        return possibleValues

    def render(self,mode='human'):
        def text(surface, fontFace, size, x, y, text, colour):
                font = pygame.font.SysFont(fontFace, size)
                text = font.render(text, 1, colour)
                surface.blit(text, (x, y))
        pygame.init()
        ratio = 1
        # Game window size
        size = self.width, self.height = int(700*ratio), int(600*ratio)
        # Game color bank
        self.blue = 0, 0, 255
        self.red = 255, 0, 0
        self.yellow = 255, 192 , 203
        self.white = 255, 255, 255
        # Setting screen
        self.screen = pygame.display.set_mode(size)
        # Start your engines
        run = True
        # Creating our exit condition
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        #background blue
        self.screen.fill(self.blue)

        #White circles:
        for i in range (0,self.boardX):
            for z in range (0,self.boardY):
                if self.state[z][i] == 0:
                    pygame.draw.circle(self.screen,self.white,(50*ratio +100*i*ratio,50*ratio+ ratio*100*z),ratio*45)
                if self.state[z][i] == 1:
                    pygame.draw.circle(self.screen,self.yellow,(50*ratio +100*i*ratio,50*ratio+ ratio*100*z),ratio*45)
                if self.state[z][i] == 2:
                    pygame.draw.circle(self.screen,self.red,(50*ratio +100*i*ratio,50*ratio+ ratio*100*z),ratio*45)

        self.winCheck()
        pygame.display.update()
        if self.yellowWin:
            text(self.screen, 'Comic Sans MS', 55, 300, 300, 'Yellow Wins!', (0, 255, 0))
            pygame.display.update()
            time.sleep(3)
            pygame.quit()
        if self.redWin:
            text(self.screen, 'Comic Sans MS', 45, 300, 300, 'Red Wins!', (0, 255, 0))
            pygame.display.update()
            time.sleep(3)
            pygame.quit()

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
        self.colour = "Yellow"
        self.reward = 5
        self.redWin = False
        self.yellowWin = False
        self.done = False
        self.turn = True
        return self.state

    def possibilities(self,colour):
        legalMoves = self.legalCheck()
        possibleStates = []
        for i in legalMoves:
            state = self.stateGen(i,self.state,colour)
            possibleStates.append([state,i])
        return possibleStates

    def stateGen(self,action,stateTest,colour):
        testState = copy.deepcopy(stateTest)
            #Yellow
        if colour:
            for i in range (self.boardY-1,-1,-1):
                if testState[i][action] == 0:
                    testState[i][action] = 1
                    return testState
            #Red
        if not colour:
            for i in range (self.boardY-1,-1,-1):
                if testState[i][action] == 0:
                    testState[i][action] = 2
                    return testState

    def winTaker(self,stateList):
        for i in stateList:
            if self.takeFour(i[0]) == 500 or self.takeFour(i[0]) == 200:
                return(i[1])
        return 9

    def takeFour(self,stateToEval):
        if stateToEval is not None:
            #Longways, breaks increase efficiency.
            connectAmount = 4
            YellowConnect = False
            RedConnect = False 

            for i in range(0,self.boardY):
                for z in range (0,self.boardX-connectAmount+1):
                    counter1 = 0
                    counter2 = 0
                    for d in range (0,connectAmount):
                        if stateToEval[i][z+d] == 1:
                            counter1 += 1 
                        if stateToEval[i][z+d] == 2:
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
                            if stateToEval[i+d][z] == 1:
                                counter1 += 1
                            if stateToEval[i+d][z] == 2:
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
                            if stateToEval[i+d][z-d] == 1:
                                counter1 += 1
                            if stateToEval[i+d][z-d] == 2:
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
                            if stateToEval[i+d][z+d] == 1:
                                counter1 += 1
                            if stateToEval[i+d][z+d] == 2:
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
                return 500
            if RedConnect:
                return 200
        else:
            return 0


done = False
env = Connect()

# =====================================================================
# 💡 自動化資料管線 (資料夾收納版) 與 互動式選單
# =====================================================================
import csv
import os
import glob
import pandas as pd
from multiprocessing import Pool
import numpy as np

# 設定資料夾名稱
DATA_DIR = "raw_data"

def f(x):
    done = False
    gameStates = []
    env = Connect()

    while not done:
        action = 2
        obs, reward, done, info = env.step(action)
        gameStates.append(obs.tolist())

    # 確保資料夾存在，並把檔案存進去 (檔名格式: raw_data/1_0.csv)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if reward == 0:
        with open(f'{DATA_DIR}/{x}_0.csv', 'a', newline='') as myfile:
            csv.writer(myfile, quoting=csv.QUOTE_ALL).writerow(gameStates)
    elif reward == 1:
        with open(f'{DATA_DIR}/{x}_1.csv', 'a', newline='') as myfile:
            csv.writer(myfile, quoting=csv.QUOTE_ALL).writerow(gameStates)
    elif reward == -1:
        with open(f'{DATA_DIR}/{x}_neg1.csv', 'a', newline='') as myfile:
            csv.writer(myfile, quoting=csv.QUOTE_ALL).writerow(gameStates)
            
    env.reset()

def process_csv_pattern(file_pattern):
    # 自動抓取資料夾內符合規則的所有碎片檔案
    files = glob.glob(file_pattern)
    all_values = []
    
    for filename in files:
        try:
            df = pd.read_csv(filename, on_bad_lines='skip', names=[i for i in range(0, 43)])
        except TypeError:
            df = pd.read_csv(filename, error_bad_lines=False, names=[i for i in range(0, 43)])
            
        for column in df:
            all_values += df[column].tolist()
            
    if not all_values:
        return np.array([])
    return pd.DataFrame(all_values).dropna().to_numpy()

def merge_and_balance_data(target_samples=400000):
    print(f"\n🧹 開始讀取並清理 [{DATA_DIR}] 資料夾內的所有 CSV 碎片檔案...")
    
    # 讀取資料夾內所有的勝、負、平手檔案
    npd1 = process_csv_pattern(f'{DATA_DIR}/*_1.csv')
    npd0 = process_csv_pattern(f'{DATA_DIR}/*_0.csv')
    npn1 = process_csv_pattern(f'{DATA_DIR}/*_neg1.csv')

    print(f"📊 目前收集的狀態數 -> 黃勝: {len(npd1)}, 平手: {len(npd0)}, 紅勝: {len(npn1)}")
    
    if len(npd1) == 0 or len(npd0) == 0 or len(npn1) == 0:
        print("⚠️ 警告：某種結局的資料量為 0！請先生成更多對弈數據。")
        return

    print("\n⚖️ 執行資料平衡 (強制 1:1:1 比例) 以強化 AI 後手防禦力...")
    min_len = min(len(npd1), len(npd0), len(npn1))
    sample_size = min(min_len, target_samples // 3)

    npd1_sampled = npd1[np.random.choice(npd1.shape[0], sample_size, replace=False)]
    npd0_sampled = npd0[np.random.choice(npd0.shape[0], sample_size, replace=False)]
    npn1_sampled = npn1[np.random.choice(npn1.shape[0], sample_size, replace=False)]

    trainingData = []
    results = []
    
    for data in npd0_sampled:
        trainingData.append(data[0])
        results.append(0)
    for data in npd1_sampled:
        trainingData.append(data[0])
        results.append(1)
    for data in npn1_sampled:
        trainingData.append(data[0])
        results.append(-1)

    print("💾 正在打包並打亂最終教材...")
    mainFrame = pd.concat([pd.DataFrame(trainingData, columns=['data']), 
                           pd.DataFrame(results, columns=['res'])], axis=1)
    
    mainFrame = mainFrame.sample(frac=1).reset_index(drop=True)
    # 最終產出的 final.csv 放在專案主目錄下供神經網路讀取
    mainFrame.to_csv('final.csv', index=False)
    print("✅ 完美！已成功產出平衡的 final.csv 教材！")

if __name__ == '__main__':
    print("=======================================")
    print("🧠 AlphaZero 訓練資料管線中心")
    print("=======================================")
    print("1. 🎲 生成對弈數據 (存入 raw_data 資料夾)")
    print("2. 🧹 清理並合併數據 (讀取 raw_data 並產出 final.csv)")
    print("=======================================")
    
    choice = input("👉 請選擇要執行的步驟 (1 或 2): ")
    
    if choice == '1':
        batches = int(input("請問要執行幾個 Batch (1 batch = 8 局，建議 1000 (原始work : 10000)): "))
        print(f"\n🚀 開始平行運算生成 {batches * 8} 局遊戲...")
        with Pool(8) as p:
            for i in range(0, batches):
                p.map(f, [1, 2, 3, 4, 5, 6, 7, 8])
                if (i + 1) % 10 == 0:
                    print(f"已完成 {(i + 1) * 8} 局...")
        print("\n🎉 生成完畢！")
        
    elif choice == '2':
        merge_and_balance_data()
    else:
        print("❌ 選擇無效，請重新執行程式。")
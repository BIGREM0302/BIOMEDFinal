import os
from os import environ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from posixpath import defpath
from numpy.core.fromnumeric import take
import pygame
import random
import copy
import time
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
from scipy import special as sp
from posixpath import defpath
import re
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
        self.yellow = 255, 255 , 0
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
            print('yellowWin')
            pygame.display.update()
            time.sleep(5)
            pygame.quit()
        if self.redWin:
            text(self.screen, 'Comic Sans MS', 45, 300, 300, 'Red Wins!', (0, 255, 0))
            print('redWin')
            pygame.display.update()
            time.sleep(5)
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
        self.colour = 1
        self.reward = 0
        self.redWin = False
        self.yellowWin = False
        self.done = False
        self.modelChoice = str(random.randint(7,14))
        return self.state

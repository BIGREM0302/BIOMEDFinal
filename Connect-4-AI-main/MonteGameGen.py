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

# counter = 0 


#We generate the games here

# def f(x):
#     done = False
#     gameStates = []
#     env = Connect()

#     while not done:
#         action = 2
#         obs, reward, done, info = env.step(action)
#         gameStates.append(obs.tolist())

#     if reward == 0:
#         with open(str(x) + '0.csv', 'a', newline='') as myfile:
#             wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#             wr.writerow(gameStates)
#     if reward == 1:
#         with open(str(x) + '1.csv', 'a', newline='') as myfile:
#             wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#             wr.writerow(gameStates)
#     if reward == -1:
#         with open(str(x) + 'neg1.csv', 'a', newline='') as myfile:
#             wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#             wr.writerow(gameStates)
#     myfile.close()
#     done = False
#     env.reset()

# if __name__ == '__main__':
#     with Pool(8) as p:
#         for i in range (0,10000):
#                 p.map(f, [1,2,3,4,5,6,7,8])


#COMBINE resulting files by hand, comment out code above, then do the following:


df = pd.read_csv('1.csv', error_bad_lines = False,names = [i for i in range(0, 43)])

all_values = []
for column in df:
    this_column_values = df[column].tolist()
    all_values += this_column_values

one_column_df = pd.DataFrame(all_values)
data1 = one_column_df.dropna()
npd1 = data1.to_numpy()

# ##

df1 = pd.read_csv('0.csv', error_bad_lines = False, names = [i for i in range(0, 43)])

all_values = []
for column in df1:
    this_column_values = df1[column].tolist()
    all_values += this_column_values

one_column_df1 = pd.DataFrame(all_values)
data0 = one_column_df1.dropna()
npd0 = data0.to_numpy()

# ###
df2 = pd.read_csv('neg1.csv', error_bad_lines = False, names = [i for i in range(0, 43)])

all_values = []
for column in df2:
    this_column_values = df2[column].tolist()
    all_values += this_column_values

one_column_df2 = pd.DataFrame(all_values)
dataNeg1 = one_column_df2.dropna()

npn1 = dataNeg1.to_numpy()


# ##

def dataChooser():

    #this is 0-2
    dataChoice = (random.randint(0,2))

    if dataChoice == 0:
        return npd0[np.random.choice(npd0.shape[0],1)][0][0] , 0 
    if dataChoice == 1:
        return npd1[np.random.choice(npd1.shape[0],1)][0][0] , 1
    if dataChoice == 2:
        return npn1[np.random.choice(npn1.shape[0],1)][0][0], -1


trainingData = []
results = []

# #We have generated 500,000 games
# #I have sampled 4,000,000 positions which is actually much smaller (1/3rd of our data)
# #Making this more could be good and wouldnt take too long.

for i in range (0,4000000):

    data,result = dataChooser()
    trainingData.append(data)
    results.append(result)
    if i % 10000 == 0:
        print (i)

df = pd.DataFrame (trainingData, columns = ['data'])
dfr = pd.DataFrame (results, columns = ['res'])

conc = [df,dfr]
mainFrame = pd.concat(conc, axis=1)

print('saving')

with open('final.csv','a') as f:
    mainFrame.to_csv(f)


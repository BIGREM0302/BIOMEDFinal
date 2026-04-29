import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from posixpath import defpath
from numpy.core.fromnumeric import take
import numpy as np
from stable_baselines3.common.utils import set_random_seed
import treelib
from treelib import Node, Tree
from scipy import special as sp
import copy
import time
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from stable_baselines3 import PPO
import torch as th
from stable_baselines3.common.utils import set_random_seed
import pandas as pd
import tf_keras as keras  
import tensorflow as tf
from tensorflow.keras import layers , models
import dask.dataframe as dd
from scipy.signal import convolve2d
from multiprocessing import Pool
from connect4 import Connect4
import tensorflow as tf

model = keras.models.load_model("monted")

def heuristics(state,colour):
    
    #Yellow is 1, red is 2
    boardX = 7
    boardY = 6

    #Give it a move, it previews the state that returns. 
    def placer(action,stateTest):
        yel = np.count_nonzero(stateTest == 1)
        red = np.count_nonzero(stateTest == 2)

        # #Stops it from affecting the original array. 
        testState = copy.deepcopy(stateTest)
        #Yellows turn 
        if yel == red:
            for i in range (boardY-1,-1,-1):
                if testState[i][action] == 0:
                    testState[i][action] = 1
                    return testState
        #Red
        else:
            for i in range (boardY-1,-1,-1):
                if testState[i][action] == 0:
                    testState[i][action] = 2
                    return testState

    #Legal move check for search. 
    def legalCheck(stateEval):
        possibleValues = []
        #Find colour using numbers, then put in newfour.
        for i in range(0,boardX):
            if stateEval[0][i] == 0:
                    possibleValues.append(i)
        return possibleValues

    #Everything already has an eval, and if the leaves eval doesnt change the parent, it stays the same. 
    #I just want the leaves of the tree to be evaled, and i want placeholder evals in the meantime.
    def treeGen(state):
        tree = Tree()
        tree.create_node(0,"00")
        counterL = 0
        start = time.process_time()
        #This creates 1 move futures
        for i in legalCheck(state):
            name = str(i)
            firstEntry = placer(int(i),state)
            tree.create_node(newFour(firstEntry,int(i)), int("1" + name), parent="00", data = firstEntry)
        #Keeps going to deeper levels if it hasnt seen at least xyz positions.
        #It will go up by multiples of 7 at the start. 
        #14000 stable
        #40000 other
        #StartL = counterL just makes sure youre actually generating information.
        #If youre not, it will break the while loop
        while counterL < 14000:
            startL = counterL
            #First we generate tree
            for node in tree.leaves():
                if int(node.tag) == 0: 
                    for i in legalCheck(node.data):
                        name = str(node.identifier) + str(i)
                        entry = placer(int(i),node.data)
                        if colour == 1:
                            tree.create_node(newFour(entry,int(i)), int(name), parent=node.identifier, data = entry)
                            counterL+=1
                        if colour == 2:
                            tree.create_node(-newFour(entry,int(i)), int(name), parent=node.identifier, data = entry)
                            counterL+=1
            if counterL == startL:
                break
        print('')
        print('Number of positions seen:' , counterL)
        print('Generating tree time:', time.process_time() - start)
        #populate leaves
        start = time.process_time()
        calculator(tree.leaves())
        print('Populating tree time:', time.process_time() - start)
        return tree

    def calculator(stateList):
        dataList = []
        IDList = {}
        counter = 0
        index=0
        for i in stateList:
            dataList.append(i.data)
            IDList[i.identifier] = index
            counter +=1 
            index+= 1

        if counter > 1:
            dataList = np.array(dataList)
            dataList = dataList.reshape(counter,6,7)
            dataList = tf.expand_dims(dataList, axis=-1)

            evalList = model.predict(dataList,steps=None,verbose=0,batch_size=20000).tolist()
            for z in stateList:
                if int(z.tag) > -500 and int(z.tag) < 300:
                    loc = IDList[z.identifier]
                    if colour == 1:
                        z.tag = evalList[loc][2] - evalList[loc][0]
                    if colour == 2:
                        z.tag = evalList[loc][0] - evalList[loc][2]

        return 'hello'

    def newFour(stateN,action):
        yel = np.count_nonzero(stateN == 1)
        red = np.count_nonzero(stateN == 2)
        
        xlim = 6
        ylim = 5

        if stateN is not None:
            xc = action
            for i in range(0,len(stateN)):
                if stateN[i][action] != 0:
                    yc = i
                    break
            yellowConnect = 500 - yel
            redConnect = -1000 + red

            if  yel != red:
                #3 on Right
                if xc+3 <= xlim:
                    if stateN[yc][xc] == 1 and stateN[yc][xc+1] == 1 and stateN[yc][xc+2] == 1 and stateN[yc][xc+3] == 1:
                        return yellowConnect
                #2 on Right, 1 on left
                if xc-1 >= 0 and xc+2 <=xlim:
                    if xc-1 >= 0 and stateN[yc][xc-1] == 1 and stateN[yc][xc+0] == 1 and stateN[yc][xc+1] == 1 and stateN[yc][xc+2] == 1:
                        return yellowConnect
                #1 on Right, 2 on left
                if xc-2 >=0 and xc+1 <= xlim:
                    if xc-2 >= 0 and stateN[yc][xc-2] == 1 and stateN[yc][xc-1] == 1 and stateN[yc][xc+0] == 1 and stateN[yc][xc+1] == 1:
                        return yellowConnect
                #0 on Right, 3 on left
                if xc-3 >= 0:
                    if stateN[yc][xc-3] == 1 and stateN[yc][xc-2] == 1 and stateN[yc][xc-1] == 1 and stateN[yc][xc+0] == 1:
                        return yellowConnect
                #3 above, none below
                if yc+3 <= ylim:
                    if stateN[yc][xc] == 1 and stateN[yc+1][xc] == 1 and stateN[yc+2][xc] == 1 and stateN[yc+3][xc] == 1:
                        return yellowConnect
                #2 above, 1 below
                if yc-1 >= 0 and yc+2 <=ylim:
                    if stateN[yc-1][xc] == 1 and stateN[yc+0][xc] == 1 and stateN[yc+1][xc] == 1 and stateN[yc+2][xc] == 1:
                        return yellowConnect
                #1 above, 2 below
                if yc-2 >=0 and yc+1 <=ylim:
                    if stateN[yc-2][xc] == 1 and stateN[yc-1][xc] == 1 and stateN[yc-0][xc] == 1 and stateN[yc+1][xc] == 1:
                        return yellowConnect
                            #0 above, 3 below
                if yc-3 >=0:
                    if stateN[yc-3][xc] == 1 and stateN[yc-2][xc] == 1 and stateN[yc-1][xc] == 1 and stateN[yc-0][xc] == 1:
                        return yellowConnect
                        #3 Right diag up, 0 right diag down.
                if xc+3 <= xlim and yc+3 <= ylim:
                    if stateN[yc][xc] == 1 and stateN[yc+1][xc+1] == 1 and stateN [yc+2][xc+2] == 1 and stateN[yc+3][xc+3] == 1:
                        return yellowConnect
            #2 Right diag up, 1 right diag down.
                if yc-1 >= 0 and xc-1 >= 0 and xc+2 <= xlim and yc+2 <=ylim:
                    if stateN[yc-1][xc-1] == 1 and stateN[yc+0][xc+0] == 1 and stateN [yc+1][xc+1] == 1 and stateN[yc+2][xc+2] == 1:
                        return yellowConnect
            #1 Right diag up, 2 right diag down.
                if yc-2 >= 0 and xc-2 >= 0 and xc+1 <= xlim and yc+1 <=ylim:
                    if stateN[yc-2][xc-2] == 1 and stateN[yc-1][xc-1] == 1 and stateN [yc+0][xc+0] == 1 and stateN[yc+1][xc+1] == 1:
                        return yellowConnect
            #0 Right diag up, 3 right diag down.
                if yc-3 >= 0 and xc-3 >= 0:
                    if  stateN[yc-3][xc-3] == 1 and stateN[yc-2][xc-2] == 1 and stateN [yc-1][xc-1] == 1 and stateN[yc+0][xc+0] == 1:
                        return yellowConnect
                #3 left diag up, 0 left diag down.
                if yc-3 >= 0 and xc+3 <=xlim:
                    if  stateN[yc][xc] == 1 and stateN[yc-1][xc+1] == 1 and stateN [yc-2][xc+2] == 1 and stateN[yc-3][xc+3] == 1:
                        return yellowConnect
            #2 left diag up, 1 left diag down.
                if yc-2 >=0 and xc-1 >= 0 and yc+1 <=ylim and xc+2<=xlim:
                    if  stateN[yc+1][xc-1] == 1 and stateN[yc-0][xc+0] == 1 and stateN [yc-1][xc+1] == 1 and stateN[yc-2][xc+2] == 1:
                        return yellowConnect
            #1 left diag up, 2 left diag down.
                if yc-1 >=0 and xc-2 >= 0 and yc+2 <= ylim and xc+1 <= xlim:
                    if  stateN[yc+2][xc-2] == 1 and stateN[yc+1][xc-1] == 1 and stateN [yc][xc] == 1 and stateN[yc-1][xc+1] == 1:
                        return yellowConnect
                            #0 left diag up, 3 left diag down.
                if xc-3 >= 0 and yc+3 <=ylim:
                    if  stateN[yc+3][xc-3] == 1 and stateN[yc+2][xc-2] == 1 and stateN [yc+1][xc-1] == 1 and stateN[yc][xc] == 1:
                        return yellowConnect       
            else:            
                #3 on Right
                if xc+3 <= xlim:
                    if stateN[yc][xc] ==2 and stateN[yc][xc+1] ==2 and stateN[yc][xc+2] ==2 and stateN[yc][xc+3] ==2:
                        return redConnect
                #2 on Right, 1 on left
                if xc-1 >= 0 and xc+2 <=xlim:
                    if xc-1 >= 0 and stateN[yc][xc-1] ==2 and stateN[yc][xc+0] ==2 and stateN[yc][xc+1] ==2 and stateN[yc][xc+2] ==2:
                        return redConnect
                #1 on Right, 2 on left
                if xc-2 >=0 and xc+1 <= xlim:
                    if xc-2 >= 0 and stateN[yc][xc-2] ==2 and stateN[yc][xc-1] ==2 and stateN[yc][xc+0] ==2 and stateN[yc][xc+1] ==2:
                        return redConnect
                #0 on Right, 3 on left
                if xc-3 >= 0:
                    if stateN[yc][xc-3] ==2 and stateN[yc][xc-2] ==2 and stateN[yc][xc-1] ==2 and stateN[yc][xc+0] ==2:
                        return redConnect
                #3 above, none below
                if yc+3 <= ylim:
                    if stateN[yc][xc] ==2 and stateN[yc+1][xc] ==2 and stateN[yc+2][xc] ==2 and stateN[yc+3][xc] ==2:
                        return redConnect
                #2 above, 1 below
                if yc-1 >= 0 and yc+2 <=ylim:
                    if stateN[yc-1][xc] ==2 and stateN[yc+0][xc] ==2 and stateN[yc+1][xc] ==2 and stateN[yc+2][xc] ==2:
                        return redConnect
                #1 above, 2 below
                if yc-2 >=0 and yc+1 <=ylim:
                    if stateN[yc-2][xc] ==2 and stateN[yc-1][xc] ==2 and stateN[yc-0][xc] ==2 and stateN[yc+1][xc] ==2:
                        return redConnect
                            #0 above, 3 below
                if yc-3 >=0:
                    if stateN[yc-3][xc] ==2 and stateN[yc-2][xc] ==2 and stateN[yc-1][xc] ==2 and stateN[yc-0][xc] ==2:
                        return redConnect
                        #3 Right diag up, 0 right diag down.
                if xc+3 <= xlim and yc+3 <= ylim:
                    if stateN[yc][xc] ==2 and stateN[yc+1][xc+1] ==2 and stateN [yc+2][xc+2] ==2 and stateN[yc+3][xc+3] ==2:
                        return redConnect
            #2 Right diag up, 1 right diag down.
                if yc-1 >= 0 and xc-1 >= 0 and xc+2 <= xlim and yc+2 <=ylim:
                    if stateN[yc-1][xc-1] ==2 and stateN[yc+0][xc+0] ==2 and stateN [yc+1][xc+1] ==2 and stateN[yc+2][xc+2] ==2:
                        return redConnect
            #1 Right diag up, 2 right diag down.
                if yc-2 >= 0 and xc-2 >= 0 and xc+1 <= xlim and yc+1 <=ylim:
                    if stateN[yc-2][xc-2] ==2 and stateN[yc-1][xc-1] ==2 and stateN [yc+0][xc+0] ==2 and stateN[yc+1][xc+1] ==2:
                        return redConnect
            #0 Right diag up, 3 right diag down.
                if yc-3 >= 0 and xc-3 >= 0:
                    if  stateN[yc-3][xc-3] ==2 and stateN[yc-2][xc-2] ==2 and stateN [yc-1][xc-1] ==2 and stateN[yc+0][xc+0] ==2:
                        return redConnect
                #3 left diag up, 0 left diag down.
                if yc-3 >= 0 and xc+3 <=xlim:
                    if  stateN[yc][xc] ==2 and stateN[yc-1][xc+1] ==2 and stateN [yc-2][xc+2] ==2 and stateN[yc-3][xc+3] ==2:
                        return redConnect
            #2 left diag up, 1 left diag down.
                if yc-2 >=0 and xc-1 >= 0 and yc+1 <=ylim and xc+2<=xlim:
                    if  stateN[yc+1][xc-1] ==2 and stateN[yc-0][xc+0] ==2 and stateN [yc-1][xc+1] ==2 and stateN[yc-2][xc+2] ==2:
                        return redConnect
            #1 left diag up, 2 left diag down.
                if yc-1 >=0 and xc-2 >= 0 and yc+2 <= ylim and xc+1 <= xlim:
                    if  stateN[yc+2][xc-2] ==2 and stateN[yc+1][xc-1] ==2 and stateN [yc][xc] ==2 and stateN[yc-1][xc+1] ==2:
                        return redConnect
                            #0 left diag up, 3 left diag down.
                if xc-3 >= 0 and yc+3 <=ylim:
                    if  stateN[yc+3][xc-3] ==2 and stateN[yc+2][xc-2] ==2 and stateN [yc+1][xc-1] ==2 and stateN[yc][xc] ==2:
                        return redConnect  
            return 0
        return 0

    def legalFour(state):
        if state is not None:
        #extract coordinates of last move, get colour of last move.
            yel = np.count_nonzero(state == 1)
            red = np.count_nonzero(state == 2)
            if yel == red:
                colour = 1
            else:
                colour = 2

            if colour == 1:
                state = np.where(state == 2,0,state)
            if colour == 2:
                state = np.where(state == 1,0,state)

            horizontal_kernel = np.array([[1, 1, 1, 1]])
            vertical_kernel = np.transpose(horizontal_kernel)
            diag1_kernel = np.eye(4, dtype=np.uint8)
            diag2_kernel = np.fliplr(diag1_kernel)
            detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]
            
            if colour == 1:
                for kernel in detection_kernels:
                    if (convolve2d(state == 1, kernel,mode='valid') == 4).any() and colour == 1:
                        return 500
            elif colour == 2:
                for kernel in detection_kernels:
                    if (convolve2d(state == 2, kernel,mode='valid') == 4).any() and colour == 2:
                        return -1000
            return 0 

    def evaluation(state):
        #Minmaxing stage.
        tree = treeGen(state)
        start = time.process_time()
        #getting from bottom of tree first.
        for node in reversed(tree.all_nodes()):
            child = node.identifier
            parent = tree.parent(child)
            if parent != None:
                #If your identifier is divisible by 2, that means youre trying to maximise because its your turn
                if len(str(child)) % 2 == 0:
                    if node.tag > parent.tag or parent.tag==0:
                        parent.tag = node.tag
                #Else, youre trying to minimise because its your opponents turn.
                else:
                    if node.tag < parent.tag or parent.tag==0:
                        parent.tag = node.tag

        bestMoveTuple = (-100000,0)
        for node in tree.is_branch("00"):
            if tree.get_node(node).tag > bestMoveTuple[0]:
                bestMoveTuple = (tree.get_node(node).tag ,tree.get_node(node).identifier)
                x = tree.get_node(node).data.reshape(1,6,7)
                x = tf.expand_dims(x, axis=-1)
        
        # tree.show( idhidden=False, line_type='ascii-emh')
        print('Eval is:', round(bestMoveTuple[0],2), "Best move is:", str(bestMoveTuple[1])[1])
        print('Minmaxing Time:', time.process_time() - start)
        return int(str(bestMoveTuple[1])[1])

    return evaluation(state)

env = Connect4()
boardX = 7
boardY = 6

done = False
env.render()

for i in range (0,10):

    while not done:
        # 判斷現在輪到誰
        if env.getColour() == 1:
            # 1 代表先手 (黃色)，現在換人類玩！
            action = int(input('\n輪到你了 (先手)！請輸入落子位置 (0-6): '))
        else:
            # 2 代表後手 (紅色)，交給 AI 思考
            print('\nAI 思考中...')
            start = time.process_time()
            action = heuristics(env.getState(), 2)
            print('AI 思考時間:', round(time.process_time() - start, 2), '秒')

        obs, rewards, done, info = env.step(action)

    env.reset()
    done=False




#tensorboard --logdir /Connect4/ 

# x = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

# action= heuristics(x,1)

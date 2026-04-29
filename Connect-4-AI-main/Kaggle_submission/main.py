import os
import sys
import numpy as np
import copy
from scipy.signal import convolve2d

# ==========================================
# 1. Kaggle 環境路徑設定
# ==========================================
KAGGLE_AGENT_PATH = "/kaggle_simulations/agent/"
if os.path.exists(KAGGLE_AGENT_PATH):
    sys.path.insert(0, KAGGLE_AGENT_PATH)
else:
    KAGGLE_AGENT_PATH = os.path.dirname(__file__)

import treelib
from treelib import Node, Tree

# ==========================================
# 2. 讀取 NumPy 權重
# ==========================================
weights_path = os.path.join(KAGGLE_AGENT_PATH, "weights.npz")
npz_file = np.load(weights_path)
W_conv1, b_conv1 = npz_file['arr_0'], npz_file['arr_1']
W_conv2, b_conv2 = npz_file['arr_2'], npz_file['arr_3']
W_dense1, b_dense1 = npz_file['arr_4'], npz_file['arr_5']
W_dense2, b_dense2 = npz_file['arr_6'], npz_file['arr_7']
W_dense3, b_dense3 = npz_file['arr_8'], npz_file['arr_9']

# ==========================================
# 3. 純 NumPy 推論引擎
# ==========================================
def conv2d_valid_relu(X, W, b):
    batch, h, w, in_c = X.shape
    kh, kw, _, out_c = W.shape
    out_h, out_w = h - kh + 1, w - kw + 1
    out = np.zeros((batch, out_h, out_w, out_c))
    
    W_flat = W.reshape(-1, out_c)
    for i in range(out_h):
        for j in range(out_w):
            region = X[:, i:i+kh, j:j+kw, :].reshape(batch, -1)
            out[:, i, j, :] = np.dot(region, W_flat) + b
    return np.maximum(0, out)

def numpy_predict(dataList):
    x = conv2d_valid_relu(dataList, W_conv1, b_conv1)
    x = conv2d_valid_relu(x, W_conv2, b_conv2)
    x = x.reshape(x.shape[0], -1)
    x = np.maximum(0, np.dot(x, W_dense1) + b_dense1)
    x = np.maximum(0, np.dot(x, W_dense2) + b_dense2)
    x = np.dot(x, W_dense3) + b_dense3
    
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return probs.tolist()

# ==========================================
# 4. 你的原始搜尋邏輯 (Heuristics)
# ==========================================
def heuristics(state, colour):
    boardX = 7
    boardY = 6

    def placer(action, stateTest):
        yel = np.count_nonzero(stateTest == 1)
        red = np.count_nonzero(stateTest == 2)
        testState = copy.deepcopy(stateTest)
        if yel == red:
            for i in range(boardY-1, -1, -1):
                if testState[i][action] == 0:
                    testState[i][action] = 1
                    return testState
        else:
            for i in range(boardY-1, -1, -1):
                if testState[i][action] == 0:
                    testState[i][action] = 2
                    return testState

    def legalCheck(stateEval):
        possibleValues = []
        for i in range(0, boardX):
            if stateEval[0][i] == 0:
                possibleValues.append(i)
        return possibleValues

    def calculator(stateList):
        dataList = []
        IDList = {}
        counter = 0
        index = 0
        for i in stateList:
            dataList.append(i.data)
            IDList[i.identifier] = index
            counter += 1 
            index += 1

        if counter > 1:
            dataList = np.array(dataList)
            # 💡 關鍵修改 1：移除 TensorFlow 的 tf.expand_dims，改用 NumPy 的 reshape
            dataList = dataList.reshape(counter, 6, 7, 1)

            # 💡 關鍵修改 2：把 model.predict 換成你的 numpy_predict
            evalList = numpy_predict(dataList)
            
            for z in stateList:
                if int(z.tag) > -500 and int(z.tag) < 300:
                    loc = IDList[z.identifier]
                    if colour == 1:
                        z.tag = evalList[loc][2] - evalList[loc][0]
                    if colour == 2:
                        z.tag = evalList[loc][0] - evalList[loc][2]
        return 'hello'

    def treeGen(state):
        tree = Tree()
        tree.create_node(0, "00")
        counterL = 0
        for i in legalCheck(state):
            name = str(i)
            firstEntry = placer(int(i), state)
            tree.create_node(newFour(firstEntry, int(i)), int("1" + name), parent="00", data=firstEntry)
            
        # 💡 關鍵修改 3：為了避免 Kaggle 8秒超時被強制判負，必須降低搜尋節點數量
        # 原本是 14000，這裡先設為 800 保證安全。如果執行速度很快，你可以慢慢往上調。
        MAX_NODES = 14000
        
        while counterL < MAX_NODES:
            startL = counterL
            for node in tree.leaves():
                if int(node.tag) == 0: 
                    for i in legalCheck(node.data):
                        name = str(node.identifier) + str(i)
                        entry = placer(int(i), node.data)
                        if colour == 1:
                            tree.create_node(newFour(entry, int(i)), int(name), parent=node.identifier, data=entry)
                            counterL += 1
                        if colour == 2:
                            tree.create_node(-newFour(entry, int(i)), int(name), parent=node.identifier, data=entry)
                            counterL += 1
                        # 防止內部迴圈超過限制
                        if counterL >= MAX_NODES:
                            break
                if counterL >= MAX_NODES:
                    break
            if counterL == startL:
                break
                
        calculator(tree.leaves())
        return tree

    def newFour(stateN, action):
        yel = np.count_nonzero(stateN == 1)
        red = np.count_nonzero(stateN == 2)
        
        xlim = 6
        ylim = 5

        # ==========================================
        # 💡 核心修改：修復 AI 後手的防禦盲區，打造鐵壁防守
        # ==========================================
        if colour == 1:
            # 當 AI 是黃色 (先手)：
            yellowConnect = 500 - yel      # 自己連線的價值 (進攻)
            redConnect = -5000 + red       # 對手連線的威脅 (防守，絕對要擋！)
        else:
            # 當 AI 是紅色 (後手)：
            # ⚠️ 注意：treeGen 呼叫時會將返回值乘上負號 (-newFour)
            yellowConnect = 5000 - yel     # 乘負號後變 -5000 (對手威脅，絕對要擋！)
            redConnect = -500 + red        # 乘負號後變 +500  (自己的進攻價值)
        # ==========================================

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

    def evaluation(state):
        tree = treeGen(state)
        for node in reversed(tree.all_nodes()):
            child = node.identifier
            parent = tree.parent(child)
            if parent != None:
                if len(str(child)) % 2 == 0:
                    if node.tag > parent.tag or parent.tag == 0:
                        parent.tag = node.tag
                else:
                    if node.tag < parent.tag or parent.tag == 0:
                        parent.tag = node.tag

        bestMoveTuple = (-100000, 0)
        # 防呆機制：萬一找不到最佳步，至少從合法步中挑一個
        valid_moves = legalCheck(state)
        if valid_moves:
            bestMoveTuple = (-100000, valid_moves[0])

        for node in tree.is_branch("00"):
            if tree.get_node(node).tag > bestMoveTuple[0]:
                bestMoveTuple = (tree.get_node(node).tag, tree.get_node(node).identifier)
        
        move_str = str(bestMoveTuple[1])
        if len(move_str) > 1:
            return int(move_str[1])
        return valid_moves[0] if valid_moves else 0

    return evaluation(state)

# ==========================================
# 5. Kaggle 官方指定進入點 (不可修改函數名稱)
# ==========================================
def agent(observation, configuration):
    """
    Kaggle 會傳入兩個參數：
    - observation.board: 長度 42 的一維陣列
    - observation.mark: 你的顏色 (1 或 2)
    """
    # 將 Kaggle 的 1D 棋盤轉換為 6x7 的 2D 矩陣
    board = np.array(observation.board).reshape(configuration.rows, configuration.columns)
    my_colour = observation.mark
    
    # 將矩陣丟進你的核心演算法
    best_action = heuristics(board, my_colour)
    
    return int(best_action)
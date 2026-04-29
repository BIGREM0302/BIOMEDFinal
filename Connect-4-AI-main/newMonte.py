import treelib
from treelib import Tree,Node
import numpy as np
import copy
import random
import time


#remove deepcopys for datahandling later.

def monte(gameState,depth):
    boardX = 7
    boardY = 6
    start = time.process_time()

    def legalCheck(stateEval):
        possibleValues = []
        if stateEval.any() != None:
            for i in range(0,boardX):
                    if stateEval[0][i] == 0:
                            possibleValues.append(i)
        return possibleValues

    def placer(givenState,action):
        yel = np.count_nonzero(givenState == 1)
        red = np.count_nonzero(givenState == 2)

        placeState = copy.deepcopy(givenState)
            #Yellow
        if yel == red:
            for i in range (boardY-1,-1,-1):
                if placeState[i][action] == 0:
                    placeState[i][action] = 1
                    return placeState
        #Red
        else:
            for i in range (boardY-1,-1,-1):
                if placeState[i][action] == 0:
                    placeState[i][action] = 2
                    return placeState

    #Making our root node.
    tree = Tree()
    visits = 0
    score = 0
    nodeValue = 0
    possibleMoves = legalCheck(gameState)

    tree.create_node(identifier=8,data=[gameState,visits,score])
    def winCheck(stateN,action):
        yel = np.count_nonzero(stateN == 1)
        red = np.count_nonzero(stateN == 2)
        
        xlim = 6
        ylim = 5
        yellowConnect = 1
        redConnect = -1

        if stateN is not None:
            xc = action
            for i in range(0,len(stateN)):
                if stateN[i][action] != 0:
                    yc = i
                    break


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
            return 0.5
        return 0.5

    def valueCalc(nVisitsChild,nScore,nVisitsParent):
        bias = 2
        #Search unexplored nodes over explored ones.
        if nVisitsChild == 0 or nVisitsParent == 0:
            answer = 100000
        else:
            answer = nScore/nVisitsChild + bias * np.sqrt (np.log(nVisitsParent) / (nVisitsChild))
        return answer

    def expansion(node):
        parent = node.identifier
        pState = node.data[0]
        if parent != 8:
            if winCheck (pState,parent%10) != 0.5:
                for i in legalCheck(pState):
                    addData = placer(pState,i)
                    tree.create_node(parent=parent,identifier=int(str(parent) + str(i)), data=[addData,0,0])
        else:
            for i in legalCheck(pState):
                addData = placer(pState,i)
                tree.create_node(parent=parent,identifier=int(str(parent) + str(i)), data=[addData,0,0])
    def selection():
        #Root
        id = 8
        while tree.is_branch(id) != []:
            values = []
            ids = []
            for i in tree.is_branch(id):
                node = tree.get_node(i)
                values.append(valueCalc(node.data[1],node.data[2],tree.parent(node.identifier).data[1]))
                ids.append(node.identifier)
            id = ids[random.choice(np.where(values == np.amax(values))[0])]
        return tree.get_node(id)

    def simulation(state,move):
        yel = np.count_nonzero(state == 1)
        red = np.count_nonzero(state == 2)

        if yel == red:
            localColour = 1
        else:
            localColour = 2

        #Terminal by result
        if winCheck(state,move) != 0.5:
            if localColour == 2:
                return winCheck(state,move)
            else:
                return -winCheck(state,move)

        if legalCheck(state) == []:
            return 0.5
        #Terminal by no moves.
        while legalCheck(state) != []:
            action = random.choice(legalCheck(state))
            state = placer(state,action)
            if winCheck(state,action) != 0.5:
                break
        
        result = winCheck(state,action)

        #Tells you whether the player with the move won or lost. 
        if localColour == 2:
            return result
        else:
            return -result

    def backProp(nodeID):
        result = simulation(tree.get_node(nodeID).data[0],tree.get_node(nodeID).identifier % 10)
        counter = 0
        node = tree.get_node(nodeID)
        #Draw:
        if result == 0.5 or result == -0.5:
            while tree.parent(nodeID) != None:
                node.data[1] = node.data[1] + 1
                node.data[2] = node.data[2] + 0.5
                node = tree.parent(node.identifier)
                nodeID = node.identifier

        #If you win, you update yourself and every second step after
        elif result == 1:
            while tree.parent(nodeID) != None:
                if counter % 2 == 0:
                    node.data[1] = node.data[1] + 1
                    node.data[2] = node.data[2] + 1
                    node = tree.parent(node.identifier)
                    nodeID = node.identifier
                else:
                    node.data[1] = node.data[1] + 1
                    node = tree.parent(node.identifier)
                    nodeID = node.identifier
                counter += 1

        #If you lose, update the one above you and every second step after. 
        elif result == -1:
            counter += 1
            while tree.parent(nodeID) != None:
                if counter % 2 == 0:
                    node.data[1] = node.data[1] + 1
                    node.data[2] = node.data[2] + 1
                    node = tree.parent(node.identifier)
                    nodeID = node.identifier
                else:
                    node.data[1] = node.data[1] + 1
                    node = tree.parent(node.identifier)
                    nodeID = node.identifier
                counter += 1

        tree.get_node(tree.root).data[1] += 1


    for i in range (0,depth):
        expansionChoice = selection()
        expansion(expansionChoice)
        #If node isnt terminal 
        if tree.is_branch(expansionChoice.identifier) != []:
            propFrom = random.choice(tree.is_branch(expansionChoice.identifier))
        #If it is terminal, we backprop directly.
        else:
            propFrom = expansionChoice.identifier
        backProp(propFrom)

    values = []
    ids = []
    scores = []
    for i in tree.is_branch(8):
        node = tree.get_node(i)
        values.append(node.data[1])
        scores.append(node.data[2])
        ids.append(node.identifier)
    id = ids[random.choice(np.where(values == np.amax(values))[0])]

    # for i in tree.all_nodes():
    #     if i.data[1] != 0:
    #         print(i.data[0],i.data[1],i.data[2],i.identifier)

    # print(scores)
    # print(values)
    # print(ids)
    # print(id)
    # print('Monte Time', time.process_time() - start)           
    return int(str(id)[1])
    # [gameState,visits,score,nodeValue,possibleMoves])

#If it selects a terminal node, what do?

test = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],[0, 0, 0, 1, 2, 0, 0], [0, 0, 0, 1, 2, 0, 0]]
test = np.array(test)

# print(test)
# monte(test,4000)
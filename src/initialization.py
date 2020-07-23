import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt

####################################################################################################
#All Functions are below:
####################################################################################################


# construct the tensor, append entries to corresponding list
def tensor_initialization(keyNumber,key_lst,prob,tensor,pt,plst,elst):
    for i in range(keyNumber):
        tensor.append([])
        pt.append([])
        for j in range(keyNumber):
            tensor[i].append([])
            pt[i].append([])
            for k in range(keyNumber):
                word = (key_lst[i]+key_lst[j]+key_lst[k])
                elst.append(word)
                Probability = prob[i]*prob[j]*prob[k]
                Probability = round(Probability,10)
                tensor[i][j].append(word)
                pt[i][j].append(Probability)


# construct the matrix
def matrix_initialization(keyNumber,key_lst,prob,matrix,pm,mplst,melst):
    for i in range(keyNumber):
        matrix.append([])
        pm.append([])
        for j in range(keyNumber):
            word = (key_lst[i]+key_lst[j])
            melst.append(word)
            Probability = round(prob[i]*prob[j],10)
            matrix[i].append(word)
            pm[i].append(Probability)


def print_tensor(ten,key):
    for i in range(key):
        for j in range(key):
            print(ten[i][j][0], ten[i][j][1], ten[i][j][2])
        print('\n')

def layerSum(ten,key,lay):
    sum = 0.0
    for i in range(key):
        for j in range(key):
            sum = sum + ten[lay][i][j]

    sum = round(sum,10)
    return sum

def rowSum(ten,key,row):
    sum = 0.0
    for i in range(key):
        for j in range(key):
            sum = sum + ten[i][row][j]
    sum = round(sum,10)
    return sum

def colSum(ten,key,col):
    sum = 0.0
    for i in range(key):
        for j in range(key):
            sum = sum + ten[i][j][col]
    sum = round(sum,10)
    return sum

def matrixRowSum(mat,key,row):
    sum = 0.0
    for i in range(key):
        sum = sum + mat[row][i]
    sum = round(sum,10)
    return sum


def imbalance(ten,key,al,be,prob,plst): #tensor, keyNumber, alpha, beta
    for i in range(key):
        for j in range(key):
            for k in range(key):
                if (i != j and j != k and i != k):#white
                    ten[i][j][k] = round(ten[i][j][k]*al*be,10)
                elif(i == j == k):
                    1
                else:
                    ten[i][j][k] = round(ten[i][j][k]*be,10)
    for i in range(key):
        ten[i][i][i] = ten[i][i][i]+(prob[i]-layerSum(ten,key,i))
    for i in range(key):
        for j in range(key):
            for k in range(key):
                plst.append(ten[i][j][k])


def imbalanceMatrix(mat,key,al,prob,mplst):#matrix, keyNumber, alpha
    for i in range(key):
        for j in range(key):
            if(i != j):
                mat[i][j] = round(mat[i][j]*al,10)
    for i in range(key):
        mat[i][i] = round(mat[i][i]+(prob[i]-matrixRowSum(mat,key,i)),10)
    for i in range(key):
        for j in range(key):
            mplst.append(mat[i][j])


#adding an edge with weights
def adding_edges(e1,e2,G):
    if G.has_edge(e1,e2):
        # we added this one before, just increase the weight by one
        G[e1][e2]['weight'] += 1
    else:
        # new edge. add with weight=1
        G.add_edge(e1,e2, weight=1)


def make(G,elst,plst,melst,mplst,key_lst,prob):
    L1 = ''
    L2 = ''
    L3 = ''
    T_L1 = True
    T_L2 = True
    T_L3 = True
    TNum = 3

    while(TNum == 3):
        lst = np.random.choice(elst,1,p=plst)[0]

        if (lst[0] == 's'):
            if (len(L1) != 0):
                L1 = L1 + lst[0]
                T_L1 = False
                TNum = TNum - 1
        else:
            L1 = L1 + lst[0]

        if (lst[1] == 's'):
            if (len(L2) != 0):
                L2 = L2 + lst[1]
                T_L2 = False
                TNum = TNum - 1
        else:
            L2 = L2 + lst[1]

        if (lst[2] == 's'):
            if (len(L3) != 0):
                L3 = L3 + lst[2]
                T_L3 = False
                TNum = TNum - 1
        else:
            L3 = L3 + lst[2]



    while(TNum == 2):
        lst = np.random.choice(melst,1,p=mplst)[0]
        if(T_L1 == False):
            if (lst[0] == 's'):
                if (len(L2) != 0):
                    L2 = L2 + lst[0]
                    T_L2 = False
                    TNum = TNum - 1
            else:
                L2 = L2 + lst[0]

            if (lst[1] == 's'):
                if (len(L3) != 0):
                    L3 = L3 + lst[1]
                    T_L3 = False
                    TNum = TNum - 1
            else:
                L3 = L3 + lst[1]



        elif(T_L2 == False):
            if (lst[0] == 's'):
                if (len(L1) != 0):
                    L1 = L1 + lst[0]
                    T_L1 = False
                    TNum = TNum - 1
            else:
                L1 = L1 + lst[0]

            if (lst[1] == 's'):
                if (len(L3) != 0):
                    L3 = L3 + lst[1]
                    T_L3 = False
                    TNum = TNum - 1
            else:
                L3 = L3 + lst[1]




        else:
            if (lst[0] == 's'):
                if (len(L1) != 0):
                    L1 = L1 + lst[0]
                    T_L1 = False
                    TNum = TNum - 1
            else:
                L1 = L1 + lst[0]

            if (lst[1] == 's'):
                if (len(L2) != 0):
                    L2 = L2 + lst[1]
                    T_L2 = False
                    TNum = TNum - 1
            else:
                L2 = L2 + lst[1]

    while(TNum == 1):
        lst = np.random.choice(key_lst,1,p=prob)[0]
        if(T_L1 == True):
            if (lst[0] == 's'):
                if (len(L1) != 0):
                    L1 = L1 + lst[0]
                    T_L1 = False
                    TNum = TNum - 1
            else:
                L1 = L1 + lst[0]
        elif(T_L2 == True):
            if (lst[0] == 's'):
                if (len(L2) != 0):
                    L2 = L2 + lst[0]
                    T_L2 = False
                    TNum = TNum - 1
            else:
                L2 = L2 + lst[0]
        else:
            if (lst[0] == 's'):
                if (len(L3) != 0):
                    L3 = L3 + lst[0]
                    T_L3 = False
                    TNum = TNum - 1
            else:
                L3 = L3 + lst[0]


    #adding edges to Graph
    if(L1!=L2):
        adding_edges(L1,L2,G)
    if(L1!=L3):
        adding_edges(L1,L3,G)
    if(L2!=L3):
        adding_edges(L2,L3,G)
    #in case all three nodes are the same
    G.add_node(L1)

    triple = []
    triple.append(L1)
    triple.append(L2)
    triple.append(L3)

    # if len(L1)>4:
    #     words_longer_than_4+=1
    # if len(L2)>4:
    #     words_longer_than_4+=1
    # if len(L3)>4:
    #     words_longer_than_4+=1
    # if L1[0]=='a':
    #     counta+=1
    # elif L1[0]=='b':
    #     countb+=1
    # elif L1[0]=='c':
    #     countc+=1
    # else:
    #     counts+=1
    #
    #
    #
    # if L2[0]=='a':
    #     counta+=1
    # elif L2[0]=='b':
    #     countb+=1
    # elif L2[0]=='c':
    #     countc+=1
    # else:
    #     counts+=1
    #
    #
    # if L3[0]=='a':
    #     counta+=1
    # elif L3[0]=='b':
    #     countb+=1
    # elif L3[0]=='c':
    #     countc+=1
    # else:
    #     counts+=1
    return triple





def generate_graph(G,triples,elst,plst,melst,mplst,key_lst,prob):
    graph = []
    for i in range(triples):
        graph.append(make(G,elst,plst,melst,mplst,key_lst,prob))
    return graph

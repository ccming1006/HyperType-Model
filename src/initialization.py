import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt

####################################################################################################
#All Functions are below:
####################################################################################################

# parameters:
#     int keyNumber, number of keys in the key list
#     lst key_lst, a list consisting all the keys
#     lst prob, a list consisting probability fo keys
#     lst(3d) tensor, a 3 dimensional list of key combinations will be made after tensor_initialization is called
#     lst(2d) pt, a probability table
#     lst plst, a probability list
#     lst elst, an entry list
# tensor_initialization constructs the tensor (ten). After running, the procedure produces a tensor with combinations of
#     keys at each entry, according to the entry's index. Probability list (plst) and entry list (elst) are also produced.

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


# parameters (above mentioned parameters will not be introduced again):
#     lst(2d) matrix, a 2 dimensional list of key combinations will be made after matrix_initialization is called
#     lst(2d) pm, probability matrix
#     lst mplst, probability list for the produced matrix
#     lst melst, entry list for the produced matrix
# matrix_initialization constructs the matrix (matrix). After running, the produre produces a matrix with combinations of two keys at
#     each entry, according to the entry's index. Probability list (mplst) and entry list (melst) are also produced.


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

# print_tensor takes a tensor and the key number, and will print the entries in the tensor.

def print_tensor(ten,keyNumber):
    for i in range(keyNumber):
        for j in range(keyNumber):
            print(ten[i][j][0], ten[i][j][1], ten[i][j][2])
        print('\n')

# layerSum takes a tensor, the key number, and the layer index. It will return the sum of probability of a layer of tensor.
def layerSum(ten,keyNumber,lay):
    sum = 0.0
    for i in range(keyNumber):
        for j in range(keyNumber):
            sum = sum + ten[lay][i][j]

    sum = round(sum,10)
    return sum

# rowSum takes a tensor, the key number, and the row index. It will sum up the probability of the desinated row for each layer, and return it.
def rowSum(ten,key,row):
    sum = 0.0
    for i in range(key):
        for j in range(key):
            sum = sum + ten[i][row][j]
    sum = round(sum,10)
    return sum

# colSum takes a tensor, the key number, and the col index. It will sum up the probability of the desinated column for each layer, and return it.
def colSum(ten,key,col):
    sum = 0.0
    for i in range(key):
        for j in range(key):
            sum = sum + ten[i][j][col]
    sum = round(sum,10)
    return sum

# colSum takes a tensor, the key number, and the row index. It will sum up the probability of the desinated row of mat, and return it.
def matrixRowSum(mat,key,row):
    sum = 0.0
    for i in range(key):
        sum = sum + mat[row][i]
    sum = round(sum,10)
    return sum

# parameters (above mentioned parameters will not be introduced again):
#     float al, an imbalance parameter from 0 to 1.
#     float be, an imbalance parameter from 0 to 1.
# imbalance will decrease type 2 and 3 entries of ten by multipling probability on type 2 and 3 entries by be, and probability of type 3 entries by al again.
#     Then type 1 entries will increase their probability by the amount that type 2 and 3 entries on their layers decreased.

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

# imbalanceMatrix will decrease type 2 entries of mat by multipling probability on type 2 entries by be.
#     Then type 1 entries will increase their probability by the amount that type 2 entries on
#     their layers decreased.
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

# parameters (above mentioned parameters will not be introduced again):
#     grah G, a graph
# make takes the probability lists and entry lists for tensor and matrix, and constructs a triple (a group of three words) based on those Probability
#     and entries. A word is terminated if an space character is appended to it. When no word terminates, it chooses entries from ten; when one word terminates,
#     it chooses entries from mat; when two words terminate, it chooses entries from prob (in this case will only choose one letter from key array); when all
#     three words terminate, the triple will be added to G and returned.
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


        L1 = L1 + lst[0]

        L2 = L2 + lst[1]

        L3 = L3 + lst[2]

        if (lst[0] == 's'):
            T_L1 = False
            TNum = TNum - 1
        if (lst[1] == 's'):
            T_L2 = False
            TNum = TNum - 1
        if (lst[2] == 's'):
            T_L3 = False
            TNum = TNum - 1



    while(TNum == 2):
        lst = np.random.choice(melst,1,p=mplst)[0]
        if(T_L1 == False):
            L2 = L2 + lst[0]
            L3 = L3 + lst[1]
            if(lst[0] == 's'):
                T_L2 = False
                TNum = TNum - 1
            if(lst[1] == 's'):
                T_L3 = False
                TNum = TNum - 1
        elif(T_L2 == False):
            L1 = L1 + lst[0]
            L3 = L3 + lst[1]
            if(lst[0] == 's'):
                T_L1 = False
                TNum = TNum - 1
            if(lst[1] == 's'):
                T_L3 = False
                TNum = TNum - 1
        else:
            L1 = L1 + lst[0]
            L2 = L2 + lst[1]
            if(lst[0] == 's'):
                T_L1 = False
                TNum = TNum - 1
            if(lst[1] == 's'):
                T_L2 = False
                TNum = TNum - 1

    while(TNum == 1):
        lst = np.random.choice(key_lst,1,p=prob)[0]
        if(T_L1 == True):
            L1 = L1 + lst[0]
        elif(T_L2 == True):
            L2 = L2 + lst[0]
        else:
            L3 = L3 + lst[0]
        if(lst[0]=='s'):
            TNum = TNum - 1


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
    return triple


# parameters (above mentioned parameters will not be introduced again):
#     int triples, number of triples to generate
# generate_graph will generate triples with make() for the desinated number of triples, and put nodes and edges into G.
# For example, new_graph=generate_graph(G,10000,elst,plst,melst,mplst,key_lst,prob) will generate a new graph.

def generate_graph(G,triples,elst,plst,melst,mplst,key_lst,prob):
    graph = []
    for i in range(triples):
        graph.append(make(G,elst,plst,melst,mplst,key_lst,prob))#
    return graph

# parameters (above mentioned parameters will not be introduced again):
#     int nodes, number of nodes to generate
# generate_graph will generate triples with make() for the desinated number of nodes, and put nodes and edges into G.
# For example, new_graph=generate_graph_nodes(G,10000,elst,plst,melst,mplst,key_lst,prob) will generate a new graph.

def generate_graph_nodes(G,nodes,elst,plst,melst,mplst,key_lst,prob):
    graph = []
    while G.order()<nodes:
        graph.append(make(G,elst,plst,melst,mplst,key_lst,prob))
    return graph

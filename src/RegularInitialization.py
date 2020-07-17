import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from initialization import *

####################################################################################################
#All Functions are below:
####################################################################################################


def make2D(G,elst,plst,melst,mplst,key_lst,prob):
    L1 = ''
    L2 = ''
    T_L1 = True
    T_L2 = True
    TNum = 2

    while(TNum == 2):
        lst = np.random.choice(melst,1,p=mplst)[0]

        L1 = L1 + lst[0]
        L2 = L2 + lst[1]

        if (lst[0] == 's'):
            T_L1 = False
            TNum = TNum - 1
        if (lst[1] == 's'):
            T_L2 = False
            TNum = TNum - 1

    while(TNum == 1):
        lst = np.random.choice(key_lst,1,p=prob)[0]
        if(T_L1 == True):
            L1 = L1 + lst[0]
        else:
            L2 = L2 + lst[0]

        if(lst[0]=='s'):
            TNum = TNum - 1

    #adding edges to Graph
    if(L1!=L2):
        adding_edges(L1,L2,G)
    #in case all three nodes are the same
    G.add_node(L1)

    pair = []
    pair.append(L1)
    pair.append(L2)

    return pair





def generate_graph2D(G,pairs,elst,plst,melst,mplst,key_lst,prob):
    graph = []
    for i in range(pairs):
        graph.append(make2D(G,elst,plst,melst,mplst,key_lst,prob))
    return graph

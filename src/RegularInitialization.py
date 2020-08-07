import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from initialization import *

# This file is to define all the functions needed in typing model
####################################################################################################
#All Functions are below:
####################################################################################################

# make2D takes the probability lists and entry lists for tensor and matrix, and constructs a pair (a group of two words) based on those Probability
#     and entries. A word is terminated if an space character is appended to it. When no word terminates, it chooses entries from mat; when one word terminates,
#     it chooses entries from prob (in this case will only choose one letter from key array); when all
#     three words terminate, the triple will be added to G and returned.
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





# generate_graph2D will generate pairs with make() for the desinated number of triples, and put nodes and edges into G.

def generate_graph2D(G,pairs,elst,plst,melst,mplst,key_lst,prob):
    graph = []
    for i in range(pairs):
        graph.append(make2D(G,elst,plst,melst,mplst,key_lst,prob))
    return graph

# generate_graph2D_nodes will generate triples with make() for the desinated number of nodes, and put nodes and edges into G.

def generate_graph2D_nodes(G,nodes,elst,plst,melst,mplst,key_lst,prob):
    graph = []
    while G.order()<nodes:
        graph.append(make2D(G,elst,plst,melst,mplst,key_lst,prob))
    return graph

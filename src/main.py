import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from initialization import *
from RegularInitialization import *

####################################################################################################
#Hand-tune Parameters
####################################################################################################
keyNumber = 4  #number of keys, including space bar
key_lst = ['a','b','c','s']
prob = [0.1,0.1,0.4,0.4] # Probability of each individual keys (3D Typing Model)
w = 110 # number of triples we want to generate
nodes = 15000 # number of nodes we want to generate
alpha = 0.9 #factor for entries with two same letters
beta = 0.95 #additional factor for entries with three different letters

####################################################################################################
#End of Hand-tune Parameters
####################################################################################################



####################################################################################################
#Global Variables
####################################################################################################
#Create a Graph
G=nx.Graph()

tensor = []# tensor_initialization() will build this tensor into a three dimensional array, in which each element is a combination of three keys.
matrix = []# matrix_initialization() will build this tensor into a two dimensional array, in which each element is a combination of two keys.
pt = []#probability tensor, a three dimensional array that stores the probability of entries in tensor.
pm = []#probability matrix, a two dimensional array that stores the probability of entries in matrix.
plst = []#probability list (tensor), an array stores all the probabilites in pt.
elst = []#entry list (tensor), an array stores all the entries in tensor.
mplst = []#probability list (matrix), an array stores all the probabilites in pm.
melst = []#entry list (matrix), an array stores all the entries in matrix.
####################################################################################################
#End of global variables
####################################################################################################


###########################Initialization for 3D Typing Model####################################
def initialization(w):
    G.clear()
    tensor.clear()
    matrix.clear()
    pt.clear()
    pm.clear()
    plst.clear()
    elst.clear()
    mplst.clear()
    melst.clear()

    #Initialize the tensor
    tensor_initialization(keyNumber,key_lst,prob,tensor,pt,plst,elst)

    #Initialize the matrix
    matrix_initialization(keyNumber,key_lst,prob,matrix,pm,mplst,melst)

    #multiply alpha and beta parameters to our
    imbalance(pt,keyNumber,alpha,beta,prob,plst)

    imbalanceMatrix(pm,keyNumber,beta,prob,mplst)

    new_graph=generate_graph(G,w,elst,plst,melst,mplst,key_lst,prob)
    # new_graph=generate_graph_nodes(G,nodes,elst,plst,melst,mplst,key_lst,prob)

initialization(w)

###########################Initialization for 3D Typing Model End####################################




###########################Initialization for 2D Typing Model####################################
# The following code is used only if we need to use original typing model to add edges
# parameters:
# pairs: the number of pairs we want to generate
# keyNumber: the number of keys (including space abr)
# key_lst: the list of keys
# prob: probability list for each key
def initialization2D(pairs,keyNumber,key_lst,prob):

    #we need to clear all the global variables since the parameters are different
    tensor.clear()
    matrix.clear()
    pt.clear()
    pm.clear()
    plst.clear()
    elst.clear()
    mplst.clear()
    melst.clear()


    #We only need to initialize the matrix since tensor is not used in the  original typing model
    matrix_initialization(keyNumber,key_lst,prob,matrix,pm,mplst,melst)
    imbalanceMatrix(pm,keyNumber,beta,prob,mplst)

    #add edges to graph G
    new_graph=generate_graph2D(G,pairs,elst,plst,melst,mplst,key_lst,prob)
# initialization2D(130000,keyNumber,key_lst,[0.25,0.25,0.1,0.4])
###########################Initialization for 2D Typing Model End####################################


print("Nodes: "+str(G.order())) #number of nodes
print("Edges: "+str(G.size()))  #number of edges
print('Average clustering coefficient: '+str(nx.average_clustering(G))) #Average clustering coefficient
print('Global clustering coefficient: '+str(nx.transitivity(G))) #Global clustering coefficient
print('Size of the LCC: '+str(len(max(nx.connected_components(G), key=len)))) #Largest Connected Component

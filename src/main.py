import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from initialization import *

####################################################################################################
#Parameter that require manual inputs
####################################################################################################
keyNumber = 3 #number of keys, assume they are 'a','b','c','s'.

key_lst = ['a','b','s']
prob = [0.15,0.45,0.4] # Probability of each individual keys
w = 1000000 # number of triples we want to generate
# nodes = 23000 # number of nodes we want to generate
alpha = 0.85 #blue factor
beta = 0.85 #additional factor

####################################################################################################
#End of manual inputs
####################################################################################################



####################################################################################################
#Global Variables
####################################################################################################
#Create a Graph
G=nx.Graph()

tensor = []
matrix = []
pt = []#probability tensor
pm = []#probability matrix
plst = []#probability list (tensor)
elst = []#entry list (tensor)
mplst = []#probability list (matrix)
melst = []#entry list (matrix)
####################################################################################################
#End of global variables
####################################################################################################


###########################Initialization####################################
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
    #Tensor
    tensor_initialization(keyNumber,key_lst,prob,tensor,pt,plst,elst)

    #matrix
    matrix_initialization(keyNumber,key_lst,prob,matrix,pm,mplst,melst)

    imbalance(pt,keyNumber,alpha,beta,prob,plst)

    imbalanceMatrix(pm,keyNumber,beta,prob,mplst)

    new_graph=generate_graph(G,w,elst,plst,melst,mplst,key_lst,prob)
    # print(new_graph)
    # print(G.number_of_edges())
initialization(w)

#
#
# def initialization_nodes(nodes):
#     global w
#     alpha=-math.log(keyNumber-1,prob[0])
#     #Tensor
#     initialization(0)
#     w= (int) ( ( (nodes / coefficient(1000)) ** (1/alpha) ) / 3)
#
#     print(w)
#     G.clear()
#     initialization(w)
#     # generate_graph(G,w,elst,plst,melst,mplst,key_lst,prob)
#     # print(G.nodes)
#
#
# def coefficient(w):
#     theory = []
#     theory_adjusted = []
#     empirical = []
#
#     #fixed values:
#     alpha=-math.log(keyNumber-1,prob[0])
#
#
#     i=1
#     while(i<=w):
#
#         #Theoretical value:
#         tri_theory=(3*i)**alpha
#         theory.append(tri_theory)
#
#
#         #Empirical Value
#         G.clear() #clear all the nodes and edges
#         generate_graph(G,i,elst,plst,melst,mplst,key_lst,prob)
#         empirical.append(G.number_of_nodes())
#
#         i+=1
#     ratio=(sum(empirical)/sum(theory))
#     return ratio
#
# initialization_nodes(nodes)
########################Initialization End####################################
print("Nodes: "+str(G.order()))
print("Edges: "+str(G.size()))
print('Average clustering coefficient: '+str(nx.average_clustering(G)))
print('Global clustering coefficient: '+str(nx.transitivity(G)))
# print('Number of Triangles: '+str(sum(nx.triangles(G).values())/3))




####################################################################################################
#Execute
####################################################################################################

# print(np.sum(pt))
# print(layerSum(pt,keyNumber,2))
# print(colSum(pt,keyNumber,2))
# print(rowSum(pt,keyNumber,2))

# for i in range(keyNumber):
#     for j in range(keyNumber):
#         for k in range(keyNumber):
#             plst.append(pt[i][j][k])



###########################Draw Graph#################################
# nx.draw(G, with_labels=False,node_size=5, edge_color='grey')
# nx.draw_spring(G, with_labels=False,node_size=40)
# nx.draw_planar(G, with_labels=True)
# nx.draw_random(G, with_labels=False,node_size=40)
# plt.show()
###########################Draw Graph End#################################


# reality=words_longer_than_4/(3*w)
# theory=(1-prob[-1])**4
# print(f'words_longer_than_4: {reality}' )
# print(f"theoretical value: {theory}" )

# print('words started with a:')
# print(counta/(3*w))
# print('words started with b:')
# print(countb/(3*w))
# print('words started with c:')
# print(countc/(3*w))
# print('words started with s:')
# print(counts/(3*w))


####################################################################################################
#Execute End
####################################################################################################

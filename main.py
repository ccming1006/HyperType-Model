import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from initialization import *

####################################################################################################
#Parameter that require manual inputs
####################################################################################################
keyNumber = 4 #number of keys, assume they are 'a','b','c','s'.

key_lst = ['a','b','c','s']
prob = [0.25,0.25,0.25,0.25] # Probability of each individual keys
w = 3 # number of triples we want to generate
alpha = 0.9 #blue factor
beta = 0.8 #additional factor

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
#Tensor
tensor_initialization(keyNumber,key_lst,prob,tensor,pt,plst,elst)

#matrix
matrix_initialization(keyNumber,key_lst,prob,matrix,pm,mplst,melst)

imbalance(pt,keyNumber,alpha,beta,prob,plst)

imbalanceMatrix(pm,keyNumber,alpha,prob,mplst)





new_graph=generate_graph(G,w,elst,plst,melst,mplst,key_lst,prob)
print(new_graph)
print(G.number_of_edges())
########################Initialization End####################################





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
nx.draw(G, with_labels=False,node_size=5, edge_color='grey')
# nx.draw_spring(G, with_labels=False,node_size=40)
# nx.draw_planar(G, with_labels=True)
# nx.draw_random(G, with_labels=False,node_size=40)
plt.show()
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

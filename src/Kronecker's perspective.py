import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
# This python file will generate a HyperType model in Kronecker's perspective. Given a key list and
# their probabilities, the code will conduct Kronecker product three times, and generate the graph with
# the ball-dropping approach


####################################################################################################
#Parameter that require manual inputs
####################################################################################################
key = ['a','b','s'] #list of keys
prob = [0.4,0.3,0.3] #list of probabilities
edges = 50 #number of edges to generate
####################################################################################################
#End of Hand-tune Parameters
####################################################################################################
keyNumber = len(key) #number of keys
pm = [] #probability matrix
rowSum = []
colSum = []
matrix = [] # r=1
entry=set()
G = nx.Graph()

# construct the initiator matrix
for i in range(keyNumber):
    matrix.append([])
    for j in range(keyNumber):
        matrix[i].append(key[i]+key[j])

# construct the P^2 matrix based on initiator matrix
matrix2 = [] # r=2

for i in range(keyNumber*keyNumber):
    matrix2.append([])
    for j in range(keyNumber*keyNumber):
        word1 = matrix[int(i/keyNumber)][int(j/keyNumber)]
        word2 = matrix[int(i%keyNumber)][int(j%keyNumber)]
        matrix2[i].append(word1+word2)


# construct the P^3 matrix based on initiator matrix
matrix3 = [] #r=3

for i in range(keyNumber*keyNumber*keyNumber):
    matrix3.append([])
    for j in range(keyNumber*keyNumber*keyNumber):
        word1 = matrix[int(i/(keyNumber*keyNumber))][int(j/(keyNumber*keyNumber))]
        word2 = matrix2[int(i%(keyNumber*keyNumber))][int(j%(keyNumber*keyNumber))]
        word = word1+word2
        word1 = word[:3]
        word2 = word[3:]
        # if word1[0] == 's':
        #     word1 = 's'
        # else:
        #     word1 = word1.split('s')[0]
        # if word2[0] == 's':
        #     word2 = 's'
        # else:
        #     word2 = word2.split('s')[0]
        word = word1+word2

        matrix3[i].append(word)


# construct the P^4 matrix based on initiator matrix
matrix4 = [] #r=4

def probability(word1,word2,k_lst,p_lst):
    result = 1.0
    for c in word1:
        result = p_lst[k_lst.index(c)]*result
    for c in word2:
        result = p_lst[k_lst.index(c)]*result
    return round(result,10)


for i in range(keyNumber*keyNumber*keyNumber*keyNumber):
    pm.append([])

#construct the probability matrix which will be used for the ball-dropping approach
for i in range(keyNumber*keyNumber*keyNumber*keyNumber):
    matrix4.append([])
    for j in range(keyNumber*keyNumber*keyNumber*keyNumber):
        word1 = matrix[int(i/(keyNumber*keyNumber*keyNumber))][int(j/(keyNumber*keyNumber*keyNumber))]
        word2 = matrix3[int(i%(keyNumber*keyNumber*keyNumber))][int(j%(keyNumber*keyNumber*keyNumber))]
        word = word1+word2
        word1 = word[:4]
        word2 = word[4:]
        if word1[0] == 's':
            word1 = 's'
        else:
            word1 = word1.split('s')[0]+'s'
        if word2[0] == 's':
            word2 = 's'
        else:
            word2 = word2.split('s')[0]+'s'


        word = word1+' '+word2
        if (word in entry) or (word1 == word2):
            matrix4[i].append(0)
            pm[i].append(0)
        else:
            p = probability(word1,word2,key,prob)
            matrix4[i].append(word1+' '+word2)
            pm[i].append(p)

for i in range(keyNumber*keyNumber*keyNumber*keyNumber):
    rowSum.append(round(sum(pm[i]),15))

total = sum(rowSum)

for i in range(keyNumber*keyNumber*keyNumber*keyNumber):
    sum = 0.0
    for j in range(keyNumber*keyNumber*keyNumber*keyNumber):
        sum += pm[j][i]
    colSum.append(round(sum,15))

for i in range(keyNumber*keyNumber*keyNumber*keyNumber):
    rowSum[i] = round(rowSum[i]/total,15)

for i in range(keyNumber*keyNumber*keyNumber*keyNumber):
    colSum[i] = round(colSum[i]/total,15)


def randRow():
    return (np.random.choice(keyNumber*keyNumber*keyNumber*keyNumber,1,p=rowSum)[0])

def randCol():
    return (np.random.choice(keyNumber*keyNumber*keyNumber*keyNumber,1,p=colSum)[0])

# Using the ball-dropping approach to add edges, only edges with positive degrees will be shown in the graph
for i in range(edges):
    pair = matrix4[randRow()][randCol()]
    word1 = pair[:4]
    word2 = pair[4:]
    G.add_edge(word1,word2)
    G.add_node(word1)
    G.add_node(word2)


# Show the graph
nx.draw(G, with_labels=True,node_size=10, edge_color='grey')
plt.show()

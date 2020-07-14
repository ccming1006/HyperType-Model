import numpy as np
import math
import networkx as nx
import re
import matplotlib.pyplot as plt





#Create a Graph
G=nx.Graph()

def read_data():
    f= open("CondMat.txt","r")
    fl =f.readlines()
    for line in fl:
        edges=re.split(r'\t+', line)
        G.add_edge(edges[0],edges[1][:-1])
    print("Nodes: "+str(G.order()))
    print("Edges: "+str(G.size()))
    print('Average clustering coefficient: '+str(nx.average_clustering(G)))
    print('Number of Triangles: '+str(sum(nx.triangles(G).values())/3))

read_data()

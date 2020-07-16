import numpy as np
import math
import networkx as nx
import re
import matplotlib.pyplot as plt
import copy



#Create a Graph
G=nx.Graph()
G_tri=nx.Graph()

def read_data():
    f= open("CondMat.txt","r")
    fl =f.readlines()
    for line in fl:
        edges=re.split(r'\t+', line)
        G.add_edge(edges[0],edges[1][:-1])
    cycle_lst = [c for c in nx.cycle_basis(G) if len(c)==3]

    print("Nodes: "+str(G.order()))
    print("Edges: "+str(G.size()))


    # print('Average clustering coefficient: '+str(nx.average_clustering(G)))
    # print('Global clustering coefficient: '+str(nx.transitivity(G)))
    # print('Number of Triangles: '+str(sum(nx.triangles(G).values())/3))
read_data()
def tri_data():
    f= open("CondMat.txt","r")
    fl =f.readlines()
    for line in fl:
        edges=re.split(r'\t+', line)
        G.add_edge(edges[0],edges[1][:-1])
    G_tri=copy.deepcopy(G)

    #list of triangles
    cycle_lst = [c for c in nx.cycle_basis(G) if len(c)==3]

    #lst of nodes that are part of triangles
    nodes_lst=nodes_in_triangles(cycle_lst,'nodes_lst')

    #remove all the nodes that are not part of triangles
    G_tri.remove_nodes_from(G.nodes-nodes_lst)

    print("Triangle Nodes: "+str(G_tri.order()))
    print("Triangle Edges: "+str(G_tri.size()))

    print('Average clust: '+str(nx.average_clustering(G_tri)))
    print('Global clust (transitivity): '+str(nx.transitivity(G_tri)))
    # print('Number of Triangles: '+str(sum(nx.triangles(G).values())/3))


def nodes_in_triangles(cycle_lst,mode):
    nodes_lst=[]
    num=0
    for i in cycle_lst:
        for j in i:
            if(j in nodes_lst):
                pass
            else:
                num+=1
                nodes_lst.append(j)
    if (mode=='num'):
        return num
    else:
        return nodes_lst

tri_data()

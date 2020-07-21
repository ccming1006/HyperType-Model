import numpy as np
import math
import networkx as nx
import re
import matplotlib.pyplot as plt
import copy



#Create a Graph
G=nx.Graph()
G_tri=nx.Graph()
G_nontri=nx.Graph()

####################################################################################################
#Data of the whole graph
####################################################################################################
def read_data():
    f= open("MU78.txt","r")
    f.readline()
    fl =f.readlines()
    for line in fl:
        edges=line.split()
        G.add_edge(edges[0],edges[1])

    print("Nodes: "+str(G.order()))
    print("Edges: "+str(G.size()))


    print('Average clustering coefficient: '+str(nx.average_clustering(G)))
    print('Global clustering coefficient: '+str(nx.transitivity(G)))
    print('Size of the LCC: '+str(len(max(nx.connected_components(G), key=len))))
    # print('Number of Triangles: '+str(sum(nx.triangles(G).values())/3))
# read_data()
####################################################################################################
#Data of the whole graph End
####################################################################################################


####################################################################################################
#Data of the Triangle part
####################################################################################################
def tri_data():
    f= open("MU78.txt","r")
    f.readline()
    fl =f.readlines()
    for line in fl:
        edges=line.split()
        G.add_edge(edges[0],edges[1])
    G_tri=copy.deepcopy(G)

    #list of triangles
    cycle_lst = [c for c in nx.cycle_basis(G) if len(c)==3]

    #lst of nodes that are part of triangles
    nodes_lst=nodes_in_triangles(cycle_lst,'nodes_lst')

    #remove all the nodes that are not part of triangles
    G_tri.remove_nodes_from(G.nodes-nodes_lst)

    print("\nTriangle Nodes: "+str(G_tri.order()))
    print("Triangle Edges: "+str(G_tri.size()))

    print('Average clust: '+str(nx.average_clustering(G_tri)))
    print('Global clust (transitivity): '+str(nx.transitivity(G_tri)))
    print('Size of the LCC: '+str(len(max(nx.connected_components(G_tri), key=len))))
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

# tri_data()

###################################################################################################
#Data of the Triangle part End
####################################################################################################



###################################################################################################
#Data of the Non-Triangle part
####################################################################################################
def nontri_data():
    f= open("MU78.txt","r")
    f.readline()
    fl =f.readlines()
    for line in fl:
        edges=line.split()
        G.add_edge(edges[0],edges[1])
    G_nontri=copy.deepcopy(G)

    #list of triangles
    cycle_lst = [c for c in nx.cycle_basis(G) if len(c)==3]

    #lst of nodes that are part of triangles
    nodes_lst=nodes_in_triangles(cycle_lst,'nodes_lst')

    #remove all the nodes that are part of triangles
    G_nontri.remove_nodes_from(nodes_lst)

    print("Triangle Nodes: "+str(G_nontri.order()))
    print("Triangle Edges: "+str(G_nontri.size()))

    print('Average clust: '+str(nx.average_clustering(G_nontri)))
    print('Global clust (transitivity): '+str(nx.transitivity(G_nontri)))
    print('Size of the LCC: '+str(len(max(nx.connected_components(G_nontri), key=len))))
# nontri_data()
###################################################################################################
#Data of the Non-Triangle part End
####################################################################################################

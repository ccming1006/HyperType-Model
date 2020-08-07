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
w = 10000 # number of triples we want to generate
nodes = 10000 # number of nodes we want to generate
alpha = 0.9 #factor alpha
beta = 0.95 #factor beta

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
#clear_all() is used when we clear all the global variables above
def clear_all():
    G.clear()
    tensor.clear()
    matrix.clear()
    pt.clear()
    pm.clear()
    plst.clear()
    elst.clear()
    mplst.clear()
    melst.clear()

###########################Initialization####################################
#initializa all the required variables
def initialization(keyNumber,key_lst,prob,w,nodes,alpha,beta):

    #clear
    tensor.clear()
    matrix.clear()
    pt.clear()
    pm.clear()
    plst.clear()
    elst.clear()
    mplst.clear()
    melst.clear()

    #initize the tensor
    tensor_initialization(keyNumber,key_lst,prob,tensor,pt,plst,elst)

    #initize the matrix
    matrix_initialization(keyNumber,key_lst,prob,matrix,pm,mplst,melst)

    #community algorithm: increase the probability of choosing entries with same letters for tensor
    imbalance(pt,keyNumber,alpha,beta,prob,plst)

    #community algorithm: increase the probability of choosing entries with same letters for matrix
    imbalanceMatrix(pm,keyNumber,beta,prob,mplst)


    # We use funtion generate_graph when we want to generate desinated number of triples
    # new_graph=generate_graph(G,w,elst,plst,melst,mplst,key_lst,prob)

    # We use funtion generate_graph_nodes when we want to generate desinated number of nodes
    new_graph=generate_graph_nodes(G,nodes,elst,plst,melst,mplst,key_lst,prob)

    return new_graph
########################Initialization End####################################





####################################################################################################
#Parameter Testing
####################################################################################################

########################Legend: the num of keys####################################
#keys() funtion is to generate plots showing the change of ACC or GCC with different number of keys and probabilities for space
#k1,k2,k3,k4 are the number of non-space keys we want to test
#   Mode a is Global clustering coefficient
#   Mode b is Average clustering coefficient
def keys(k1,k2,k3,k4,mode):
    k1_list = []
    k2_list = []
    k3_list = []
    k4_list = []

    x = range(5, 50, 5)
    s_range = [n/100 for n in x]

    k1_list=plot_lst(k1,s_range,mode)
    k2_list=plot_lst(k2,s_range,mode)
    k3_list=plot_lst(k3,s_range,mode)
    k4_list=plot_lst(k4,s_range,mode)


    plt.plot(s_range, k1_list, "b", label="k= "+ str(k1))
    plt.plot(s_range, k1_list, "b+")
    plt.plot(s_range, k2_list, "r", label="k= "+ str(k2))
    plt.plot(s_range, k2_list, "r+")
    plt.plot(s_range, k3_list, "g", label="k= "+ str(k3))
    plt.plot(s_range, k3_list, "g+")
    plt.plot(s_range, k4_list, "y", label="k= "+ str(k4))
    plt.plot(s_range, k4_list, "y+")

    # plt.title("w= "+str(w)+"; "+"alpha, beta= "+str(alpha)+", "+str(beta)+"\n Equiprobable non-space keys")
    plt.title("nodes= "+str(nodes)+"; "+"alpha, beta= "+str(alpha)+", "+str(beta)+"\n Equiprobable non-space keys")
    plt.xlabel("P(s):q")
    if (mode=='a'):
        plt.ylim(0.7, 1.0)
        plt.ylabel("Average CC")
        plt.legend(loc="lower left")
        plt.savefig("KeyNum,ACC.png")
    else:
        plt.ylim(0.0, 0.4)
        plt.legend(loc="upper right")
        plt.ylabel("Global CC")
        plt.savefig("KeyNum,GCC.png")



#funtion plot_lst() will return the list of ACC(GCC) of a specific number of keys with different q in s_range
def plot_lst(key,s_range,mode):
    global keyNumber
    keyNumber=key+1
    key_lst.clear()

    lst=[]

    for num in range(keyNumber-1):
        key_lst.append(chr(97+num))

    key_lst.append('s')
    for x in s_range:
        prob.clear()

        key_prob=round((1-x)/(keyNumber-1),2)
        for num in range(keyNumber-2):
            prob.append(key_prob)
        prob.append(round((1-x-key_prob*(keyNumber-2)),2))
        prob.append(x)

        clear_all()
        initialization(keyNumber,key_lst,prob,w,nodes,alpha,beta)
        if(mode=='a'):
            lst.append(nx.average_clustering(G))
        else:
            lst.append(nx.transitivity(G))
    return lst

keys(2,3,4,5,'b')
########################Legend: the num of keys END####################################






########################Legend: Key Prob####################################
#keyProb() funtion is to generate plots showing the change of ACC or GCC with different key probabilities
#p1,p2,p3,p4 are the probability list we want to test
#   Mode a is Global clustering coefficient
#   Mode b is Average clustering coefficient
def keyProb(p1,p2,p3,p4,mode):
    k1_list = []
    k2_list = []
    k3_list = []
    k4_list = []

    x = range(5, 50, 5)
    s_range = [n/100 for n in x]

    k1_list=plot_lst_keyProb(p1,s_range,mode)
    k2_list=plot_lst_keyProb(p2,s_range,mode)
    k3_list=plot_lst_keyProb(p3,s_range,mode)
    k4_list=plot_lst_keyProb(p4,s_range,mode)


    plt.plot(s_range, k1_list, "b", label="[P(a),P(b)]= "+ str(p1))
    plt.plot(s_range, k1_list, "b+")
    plt.plot(s_range, k2_list, "r", label="[P(a),P(b)]= "+ str(p2))
    plt.plot(s_range, k2_list, "r+")
    plt.plot(s_range, k3_list, "g", label="[P(a),P(b)]= "+ str(p3))
    plt.plot(s_range, k3_list, "g+")
    plt.plot(s_range, k4_list, "y", label="[P(a),P(b)]= "+ str(p4))
    plt.plot(s_range, k4_list, "y+")


    plt.title("nodes= "+str(nodes)+"; "+"alpha, beta= "+str(alpha)+", "+str(beta)+"\n keys: a,b,c,s")
    plt.xlabel("P(s):q")
    plt.ylabel("Average CC")
    if (mode=='a'):
        plt.ylim(0.7, 1.0)
        plt.legend(loc="lower left")
        plt.ylabel("Average CC")
        plt.savefig("KeyProb,ACC.png")
    else:
        plt.ylim(0.0, 0.2)
        plt.legend(loc="upper right")
        plt.ylabel("Global CC")
        plt.savefig("KeyProb,GCC.png")


#funtion plot_lst_keyProb() will return the list of ACC(GCC) of a specific prbability list with different q in s_range
def plot_lst_keyProb(prob_lst,s_range,mode):
    lst=[]
    for x in s_range:
        prob.clear()
        prob.append(prob_lst[0])
        prob.append(prob_lst[1])
        prob.append(1-prob_lst[0]-prob_lst[1]-round(x,2))
        prob.append(round(x,2))
        clear_all()
        initialization(keyNumber,key_lst,prob,w,nodes,alpha,beta)
        if(mode=='a'):
            lst.append(nx.average_clustering(G))
        else:
            lst.append(nx.transitivity(G))
    return lst


# keyProb([0.25,0.25],[0.2,0.3],[0.1,0.1],[0.1,0.45],'a')
########################Legend: Key Prob END####################################




########################Legend: Alpha, Beta####################################
#keyProb() funtion is to generate plots showing the change of ACC or GCC with different alpha and beta
#BetaProb1,BetaProb2,BetaProb3,BetaProb4 are the beta probabilities we want to test
#   Mode a is Global clustering coefficient
#   Mode b is Average clustering coefficient
def AlBe(BetaProb1,BetaProb2,BetaProb3,BetaProb4,mode):
    k1_list = []
    k2_list = []
    k3_list = []
    k4_list = []

    x = range(5, 105, 5)
    alpha_lst = [n/100 for n in x]

    k1_list=plot_lst_AlBe(BetaProb1,alpha_lst,mode)
    k2_list=plot_lst_AlBe(BetaProb2,alpha_lst,mode)
    k3_list=plot_lst_AlBe(BetaProb3,alpha_lst,mode)
    k4_list=plot_lst_AlBe(BetaProb4,alpha_lst,mode)


    plt.plot(alpha_lst, k1_list, "b", label="P= "+ str(BetaProb1[1]) + "Beta= "+str(BetaProb1[0]))
    plt.plot(alpha_lst, k1_list, "b+")
    plt.plot(alpha_lst, k2_list, "r", label="P= "+ str(BetaProb2[1]) + "Beta= "+str(BetaProb2[0]))
    plt.plot(alpha_lst, k2_list, "r+")
    plt.plot(alpha_lst, k3_list, "g", label="P= "+ str(BetaProb3[1]) + "Beta= "+str(BetaProb3[0]))
    plt.plot(alpha_lst, k3_list, "g+")
    plt.plot(alpha_lst, k4_list, "y", label="P= "+ str(BetaProb4[1]) + "Beta= "+str(BetaProb4[0]))
    plt.plot(alpha_lst, k4_list, "y+")

    plt.title("nodes= "+str(nodes))
    plt.xlabel("Alpha Probability")
    if (mode=='a'):
        plt.legend(loc="upper left")
        plt.ylim(0.6, 1.0)
        plt.ylabel("Average CC")
        plt.savefig("AlphaBeta,ACC.png")
    else:
        plt.legend(loc="upper left")
        plt.ylim(0.0, 0.2)
        plt.ylabel("Global CC")
        plt.savefig("AlphaBeta,GCC.png")

#funtion plot_lst_AlBe() will return the list of ACC(GCC) of a specific beta with different alphas in alpha_lst
def plot_lst_AlBe(BetaProb,alpha_lst,mode):
    lst=[]
    for alpha in alpha_lst:
        clear_all()
        initialization(keyNumber,key_lst,BetaProb[1],w,nodes,alpha,BetaProb[0])
        if(mode=='a'):
            lst.append(nx.average_clustering(G))
        else:
            lst.append(nx.transitivity(G))

    return lst


# AlBe([0.8,[0.1,0.2,0.3,0.4]],[0.95,[0.1,0.2,0.3,0.4]],[0.8,[0.25,0.25,0.25,0.25]],[0.95,[0.25,0.25,0.25,0.25]],'a')
########################Legend: Key Prob END####################################

####################################################################################################
#
####################################################################################################

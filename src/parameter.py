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
alpha = 0.9 #blue factor
beta = 0.95
 #additional white factor

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
def initialization(keyNumber,key_lst,prob,w,alpha,beta):

    #clear
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

    return new_graph
# print(new_graph)
########################Initialization End####################################





####################################################################################################
#Parameter Testing
####################################################################################################

########################Legend: the num of keys####################################

#Mode a is GCC
#Mode b is Average CC
def keys(k1,k2,k3,k4,mode):
    k1_list = []
    k2_list = []
    k3_list = []
    k4_list = []

    x = range(5, 90, 5)
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
    plt.legend(loc="upper left")
    plt.ylim(0.0, 1.0)
    plt.title("w= "+str(w)+"; "+"alpha, beta= "+str(alpha)+", "+str(beta)+"\n Equiprobable non-space keys")
    plt.xlabel("P(s):q")
    plt.ylabel("Average CC")
    if (mode=='a'):
        plt.ylabel("Average CC")
        plt.savefig("KeyNum,ACC.png")
    else:
        plt.ylabel("Global CC")
        plt.savefig("KeyNum,GCC.png")





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
        initialization(keyNumber,key_lst,prob,w,alpha,beta)
        if(mode=='a'):
            lst.append(nx.average_clustering(G))
        else:
            lst.append(nx.transitivity(G))
    return lst

keys(2,3,4,5,'b')
########################Legend: the num of keys END####################################






########################Legend: Key Prob####################################
#Mode a is GCC
#Mode b is Average CC
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

    plt.ylim(0.0, 1.0)
    plt.title("w= "+str(w)+"; "+"alpha, beta= "+str(alpha)+", "+str(beta)+"\n keys: a,b,c,s")
    plt.xlabel("P(s):q")
    plt.ylabel("Average CC")
    if (mode=='a'):
        plt.legend(loc="lower left")
        plt.ylabel("Average CC")
        plt.savefig("KeyProb,ACC.png")
    else:
        plt.legend(loc="upper left")
        plt.ylabel("Global CC")
        plt.savefig("KeyProb,GCC.png")


def plot_lst_keyProb(prob_lst,s_range,mode):
    lst=[]
    for x in s_range:
        prob.clear()
        prob.append(prob_lst[0])
        prob.append(prob_lst[1])
        prob.append(1-prob_lst[0]-prob_lst[1]-round(x,2))
        prob.append(round(x,2))
        clear_all()
        initialization(keyNumber,key_lst,prob,w,alpha,beta)
        if(mode=='a'):
            lst.append(nx.average_clustering(G))
        else:
            lst.append(nx.transitivity(G))
    return lst


# keyProb([0.25,0.25],[0.2,0.3],[0.1,0.1],[0.1,0.45],'b')
########################Legend: Key Prob END####################################




########################Legend: Alpha, Beta####################################
#Mode a is GCC
#Mode b is Average CC
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
    plt.legend(loc="upper left")
    plt.ylim(0.0, 1.0)
    plt.title("w= "+str(w))
    plt.xlabel("Alpha Probability")
    if (mode=='a'):
        plt.ylabel("Average CC")
        plt.savefig("AlphaBeta,ACC.png")
    else:
        plt.ylabel("Global CC")
        plt.savefig("AlphaBeta,GCC.png")


def plot_lst_AlBe(BetaProb,alpha_lst,mode):
    lst=[]
    for alpha in alpha_lst:
        clear_all()
        initialization(keyNumber,key_lst,BetaProb[1],w,alpha,BetaProb[0])
        if(mode=='a'):
            lst.append(nx.average_clustering(G))
        else:
            lst.append(nx.transitivity(G))

    return lst


# AlBe([0.8,[0.1,0.2,0.3,0.4]],[0.95,[0.1,0.2,0.3,0.4]],[0.8,[0.25,0.25,0.25,0.25]],[0.95,[0.25,0.25,0.25,0.25]],'b')
########################Legend: Key Prob END####################################

####################################################################################################
#
####################################################################################################

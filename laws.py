import numpy as np
from scipy import linalg as li
import math
import networkx as nx
import matplotlib.pyplot as plt
from initialization import *

####################################################################################################
#Parameter that require manual inputs
####################################################################################################
keyNumber = 4 #number of keys, assume they are 'a','b','c','s'.

key_lst = ['a','b','c','s']
prob = [0.1,0.3,0.2,0.4] # Probability of each individual keys
w = 10000 # number of triples we want to generate
alpha = 0.9 #blue factor
beta = 0.95 #additional white factor

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

imbalance(pt,keyNumber,alpha,beta,prob)

imbalanceMatrix(pm,keyNumber,beta,prob)

new_graph=generate_graph(G,w,elst,plst,melst,mplst,key_lst,prob)
# print(new_graph)
########################Initialization End####################################




####################################################################################################
#Lemma1 Testing
####################################################################################################

#Plotting
def plot_Lemma1():
    theory = []
    theory_adjusted = []
    empirical = []


    #fixed values:
    alpha=-math.log(keyNumber-1,prob[0])


    i=1
    while(i<=w):

        #Theoretical value:
        tri_theory=(3*i)**alpha
        theory.append(tri_theory)


        #Empirical Value
        G.clear() #clear all the nodes and edges
        generate_graph(G,i,elst,plst,melst,mplst,key_lst,prob)
        empirical.append(G.number_of_nodes())

        i+=1


    coeffient=0.729
    theory_adjusted = [x / coeffient for x in theory]
    ratio=(sum(theory)/sum(empirical))
    print(ratio)

    plt.figure()

    plt.subplot(111)
    plt.plot(theory)

    plt.subplot(111)
    plt.plot(theory_adjusted)
    #plt.savefig("Lemma2Theory.png")

    plt.subplot(111)
    plt.plot(empirical)
    plt.title(f"Empirical VS Theoretical p={prob[0]} q={prob[-1]} \n coefficient used for adjustment={coeffient:.3f} real_ratio={ratio:.3f}" )
    plt.savefig("Lemma1_testing.png")


#plot_Lemma1()
####################################################################################################
#Lemma1 Testing End
####################################################################################################





####################################################################################################
#Lemma2 Testing
####################################################################################################

#Plotting
def plot_Lemma2():
    theory = []
    theory_adjusted = []
    empirical = []


    #fixed values:
    alpha=-math.log(keyNumber-1,prob[0])
    c=prob[-1]**(alpha)

    i=1
    while(i<=w):

        #Theoretical value:
        n=math.log(3*i,prob[0])
        tri_theory=(3*i)**alpha*(1+(c-c**2*math.log(prob[-1],prob[0]))*n-n*(n*n+2*n-1)*c*c/2)
        theory.append(tri_theory)


        #Empirical Value
        G.clear() #clear all the nodes and edges
        generate_graph(G,i,elst,plst,melst,mplst,key_lst,prob)
        triangle_lst=nx.triangles(G)
        triangle_num=sum(triangle_lst.values())/3
        empirical.append(triangle_num)

        i+=1

    coeffient=1.609
    theory_adjusted = [x / coeffient for x in theory]
    ratio=(sum(theory)/sum(empirical))
    print(ratio)

    plt.figure()

    plt.subplot(111)
    plt.plot(theory)

    plt.subplot(111)
    plt.plot(theory_adjusted)
    #plt.savefig("Lemma2Theory.png")

    plt.subplot(111)
    plt.plot(empirical)
    plt.title(f"Empirical VS Theoretical \n p={prob[0]} q={prob[-1]} coefficient used for adjustment={coeffient}" )
    plt.savefig("Lemma2_testing.png")

#plot_Lemma2()

####################################################################################################
#Lemma2 Testing End
####################################################################################################



####################################################################################################
#L01: Degree Distribution
####################################################################################################
# reference: http://snap.stanford.edu/class/cs224w-2012/nx_tutorial.pdf
def plot_degree_distribution():
    clustering_coe=nx.average_clustering(G)
    degs = {}
    for n in G.nodes():
        deg = G.degree(n)
        if deg not in degs:
            degs[deg] = 1
        else:
            degs[deg] += 1
    items = sorted(degs.items(), key=lambda x: x[1], reverse=True)

    #print(items[:20])


    x=[math.log(x+1,10) for x in range(len(items))]
    y=[math.log(v,10) for (k,v) in items]
    plt.scatter( x , y , s=1, marker='o')

    m, b = np.polyfit(x, y, 1)
    plt.plot(x, [(m*a + b) for a in x], color='red')

    plt.ylabel("count(log10(y))")
    plt.xlabel("degree ranking(log10(x))")
    # plt.text(3, 3., f"{m}*x+{b}=y",family="serif")

    # ax.scatter( [range(len(items))],[v for (k,v) in items],s=1, marker='o')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    if (w==1000000):
        plt.title("3D Degree Distri " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+" triples: 1M")
    else:
        plt.title("3D Degree Distri " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+f" triples: {w}")

    plt.savefig("Degree Distribution testing.png")

#plot_degree_distribution()

####################################################################################################
#End of L01: Degree Distribution
####################################################################################################


####################################################################################################
#L02: Densification Power Law (DPL)
####################################################################################################

#Plotting
def plot_DPL():
    Nodes = []
    Edges = []

    i=1
    while(i<=w):

        #Empirical Value
        G.clear() #clear all the nodes and edges
        generate_graph(G,i,elst,plst,melst,mplst,key_lst,prob)
        Nodes.append(G.number_of_nodes())
        Edges.append(G.number_of_edges())

        i+=1


    x=[math.log(n,10) for n in Nodes]
    y=[math.log(e,10) for e in Edges]
    plt.scatter( x , y , s=1, marker='o')

    m, b = np.polyfit(x, y, 1)
    plt.plot(x, [(m*a + b) for a in x], color='red')

    plt.ylabel("|E|(log10(y))")
    plt.xlabel("|N|(log10(x))")

    if (w==1000000):
        plt.title("DPL(L02) " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+" triples: 1M")
    else:
        plt.title("DPL(L02) " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+f" triples: {w}")

    plt.savefig("DPL(L02).png")


#plot_DPL()
####################################################################################################
#End of L02: Densification Power Law (DPL)
####################################################################################################



####################################################################################################
#L03: Weight Power Law (WPL)
####################################################################################################

#Plotting
def plot_WPL():
    Weight = []
    Edges = []

    i=1
    while(i<=w):

        #Empirical Value
        G.clear() #clear all the nodes and edges
        generate_graph(G,i,elst,plst,melst,mplst,key_lst,prob)
        Weight.append(G.size(weight='weight'))
        Edges.append(G.number_of_edges())

        i+=1


    x=[math.log(e,10) for e in Edges]
    y=[math.log(w,10) for w in Weight]
    plt.scatter(x , y , s=1, marker='o')

    m, b = np.polyfit(x, y, 1)
    plt.plot(x, [(m*a + b) for a in x], color='red')

    plt.ylabel("|W|(log10(y))")
    plt.xlabel("|E|(log10(x))")

    if (w==1000000):
        plt.title("WPL(L03) " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+" triples: 1M")
    else:
        plt.title("WPL(L03) " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+f" triples: {w}")

    plt.savefig("WPL(L03).png")


#plot_WPL()
####################################################################################################
#End of L03: Weight Power Law (WPL)
####################################################################################################




####################################################################################################
#L04: Snapshot Power Law (SPL)
####################################################################################################

#Plotting
def plot_SPL():
    weights = {}
    for n in G.nodes():
        weight=0
        for node in G[n]:
            weight+=G[n][node]['weight']

        if weight not in weights:
            weights[weight] = 1
        else:
            weights[weight] += 1
    items = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    x=[math.log(x+1,10) for x in range(len(items))]
    y=[math.log(v,10) for (k,v) in items]
    plt.scatter( x , y , s=1, marker='o')

    m, b = np.polyfit(x, y, 1)
    plt.plot(x, [(m*a + b) for a in x], color='red')

    plt.ylabel("count(log10(y))")
    plt.xlabel("weight ranking(log10(x))")

    if (w==1000000):
        plt.title("SPL(L04) " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+" triples: 1M")
    else:
        plt.title("SPL(L04) " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+f" triples: {w}")

    plt.savefig("SPL(L04).png")


#plot_SPL()
####################################################################################################
#End of L04: Snapshot Power Law (SPL)
####################################################################################################




####################################################################################################
#L05: Triangle Power Law (TPL)
####################################################################################################

#Plotting
def plot_TPL():
    triangles = {}
    for n in G.nodes():
        tri=nx.triangles(G,n)
        if tri not in triangles:
            triangles[tri] = 1
        else:
            triangles[tri] += 1
    items = sorted(triangles.items(), key=lambda x: x[1], reverse=True)

    x=[math.log(x+1,10) for x in range(len(items))]
    y=[math.log(v,10) for (k,v) in items]
    plt.scatter( x , y , s=1, marker='o')

    m, b = np.polyfit(x, y, 1)
    plt.plot(x, [(m*a + b) for a in x], color='red')

    plt.ylabel("count(log10(y))")
    plt.xlabel("triangles(log10(x))")

    if (w==1000000):
        plt.title("TPL(L05) " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+" triples: 1M")
    else:
        plt.title("TPL(L05) " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+f" triples: {w}")

    plt.savefig("TPL(L05).png")


#plot_TPL()
####################################################################################################
#End of L05: Triangle Power Law (TPL)
####################################################################################################



####################################################################################################
#L06: Eigenvalue Power Law (EPL)
####################################################################################################

#Plotting
def plot_EPL():
    L = nx.normalized_laplacian_matrix(G)
    e = li.eigvals(L.A)
    e= [value.real for value in e]
    e = [-i if i < 0 else i for i in e]
    print(e)

    items = sorted(e)

    x=[math.log(x+1,10) for x in range(len(items))]
    y=[math.log(v,10) for v in items]
    plt.scatter( x , y , s=1, marker='o')

    m, b = np.polyfit(x, y, 1)
    plt.plot(x, [(m*a + b) for a in x], color='red')

    plt.ylabel("count(log10(y))")
    plt.xlabel("triangles(log10(x))")

    if (w==1000000):
        plt.title("EPL(L06) " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+" triples: 1M")
    else:
        plt.title("EPL(L06) " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+f" triples: {w}")

    plt.savefig("EPL(L06).png")


plot_EPL()
####################################################################################################
#End of L06: Eigenvalue Power Law (EPL)
####################################################################################################

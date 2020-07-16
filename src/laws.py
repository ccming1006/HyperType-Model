import numpy as np
from scipy import linalg as li
import math
import networkx as nx
import matplotlib.pyplot as plt
import powerlaw
from initialization import *

####################################################################################################
#Parameter that require manual inputs
####################################################################################################
keyNumber = 4 #number of keys, assume they are 'a','b','c','s'.

key_lst = ['a','b','c','s']
prob = [0.1,0.2,0.3,0.4] # Probability of each individual keys
w = 1000000 # number of triples we want to generate
words=3*w
alpha = 0.9 #blue factor
beta = 0.85 #additional white factor

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

imbalanceMatrix(pm,keyNumber,beta,prob,mplst)

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


    coefficient=1.373
    theory_adjusted = [x*coefficient for x in theory]
    x=range(0,words,3)
    ratio=(sum(empirical)/sum(theory))
    print(f"{ratio:.3f}")

    plt.figure()

    plt.subplot(111)
    plt.plot(x,theory)

    plt.plot(x,empirical)
    #plt.savefig("Lemma2Theory.png")

    plt.plot(x,theory_adjusted)
    plt.ylabel("Nodes")
    plt.xlabel("Words")

    plt.title(f"Empirical VS Theoretical p={prob[0]} q={prob[-1]} \n coefficient used for adjustment={coefficient:.3f} real ratio={ratio:.3f}" )
    plt.savefig("Lemma1_testing.png")


# plot_Lemma1()
####################################################################################################
#Lemma1 Testing End
####################################################################################################





####################################################################################################
#Lemma2 Testing
####################################################################################################
#Plotting
def plot_Lemma2():
    words=3*w
    theory = []
    theory_1 = []
    theory_2 = []
    theory_adjusted = []
    empirical = []



    #fixed values:
    alpha=-math.log(keyNumber-1,prob[0])
    c=prob[-1]**(alpha)

    i=1
    while(i<=w):

        #Theoretical value:
        n=math.log(3*i,prob[0])
        tri_theory_1=((3*i)**alpha)
        tri_theory_2=(1+(c-c*c*math.log(prob[-1],prob[0]))*n-n*(n*n+2*n-1)*c*c/2)
        tri_theory=tri_theory_1*tri_theory_2
        theory_1.append(tri_theory_1)
        theory_2.append(tri_theory_2)
        theory.append(tri_theory)


        #Empirical Value
        G.clear() #clear all the nodes and edges
        generate_graph(G,i,elst,plst,melst,mplst,key_lst,prob)
        triangle_lst=nx.triangles(G)
        triangle_num=sum(triangle_lst.values())/3
        empirical.append(triangle_num)

        i+=1
    #
    # print(theory_1)
    # print(theory_2)
    # print(theory)
    # x=range(0,words,3)
    # coefficient=(sum(empirical)/sum(theory))
    # theory_adjusted = [x*coefficient for x in theory]
    # ratio=(sum(empirical)/sum(theory))
    # print(ratio)
    #
    # plt.figure()
    #
    #
    # plt.plot(x,theory)
    #
    #
    # plt.plot(x,empirical)
    #
    # plt.plot(x,theory_adjusted)
    # plt.ylabel("Triangles")
    # plt.xlabel("Words")
    #
    # plt.title(f"Empirical VS Theoretical p={prob[0]} q={prob[-1]} \n coefficient used for adjustment={coefficient:.3f} real ratio={ratio:.3f}" )
    # plt.savefig("Lemma2_3000.png")

# plot_Lemma2()

####################################################################################################
#Lemma2 Testing End
####################################################################################################





####################################################################################################
#L01: Degree Distribution
####################################################################################################
# reference: http://snap.stanford.edu/class/cs224w-2012/nx_tutorial.pdf
def plot_degree_distribution(mode):
    clustering_coe=nx.average_clustering(G)
    degs = {}
    for n in G.nodes():
        deg = G.degree(n)
        if deg not in degs:
            degs[deg] = 1
        else:
            degs[deg] += 1

    items = sorted(degs.items())

    if(mode==1):
        file1 = open("L01Data.txt","a+")
        file1.seek(0)
        file1.truncate()
        for i in degs.values():
            file1.write(str(i)+"\n")
        file1.close()

    if(mode==2):
        data=list(degs.values())
        fit = powerlaw.Fit(data)
        R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
        print(R,p)

    else:
        items_1=[]
        items_2=[]

        for (k,v) in items:
            if v>2:
                items_1.append((k,v))
            else:
                items_2.append((k,v))



        x_1=[math.log(k+1,10) for (k,v) in items_1]
        y_1=[math.log(v,10) for (k,v) in items_1]
        plt.scatter( x_1 , y_1 , s=1, marker='o',color="red")

        x_2=[math.log(k+1,10) for (k,v) in items_2]
        y_2=[math.log(v,10) for (k,v) in items_2]
        plt.scatter( x_2 , y_2 , s=1, marker='o',color="blue")

        m, b = np.polyfit(x_1, y_1, 1)
        plt.plot(x_1, [(m*a + b) for a in x_1], color='red')

        plt.ylabel("count(log10(y))")
        plt.xlabel("degree(log10(x))")
        # plt.text(3, 3., f"{m}*x+{b}=y",family="serif")

        # ax.scatter( [range(len(items))],[v for (k,v) in items],s=1, marker='o')
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        if (w==1000000):
            plt.title("3D Degree Distri " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+" triples: 1M")
        else:
            plt.title("3D Degree Distri " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+f" triples: {w}")

        plt.savefig("Degree Distribution testing.png")

# plot_degree_distribution(3)

####################################################################################################
#End of L01: Degree Distribution
####################################################################################################


####################################################################################################
#L02: Densification Power Law (DPL)
####################################################################################################

#Plotting
def plot_DPL(mode):
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

    if (mode==1):
        fit = powerlaw.Fit(Edges)
        R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
        print(R,p)


    else:
        x=[math.log(n,10) for n in Nodes]
        y=[math.log(e,10) for e in Edges]
        plt.scatter( x , y , s=1, marker='o')

        m, b = np.polyfit(x, y, 1)
        plt.plot(x, [(m*a + b) for a in x], color='red')

        plt.ylabel("|E|(log10(y))")
        plt.xlabel("|N|(log10(x))")

        if (w==1000000):
            plt.title("DPL " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+" triples: 1M")
        else:
            plt.title("DPL " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+f" triples: {w}")

        plt.savefig("DPL(L02).png")


plot_DPL(2)
####################################################################################################
#End of L02: Densification Power Law (DPL)
####################################################################################################



####################################################################################################
#L03: Weight Power Law (WPL)
####################################################################################################

#Plotting
def plot_WPL(mode):
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

    if (mode==1):
        fit = powerlaw.Fit(Fit)
        R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
        print(R,p)

    else:
        x=[math.log(e,10) for e in Edges]
        y=[math.log(w,10) for w in Weight]
        plt.scatter(x , y , s=1, marker='o')

        m, b = np.polyfit(x, y, 1)
        plt.plot(x, [(m*a + b) for a in x], color='red')

        plt.ylabel("|W|(log10(y))")
        plt.xlabel("|E|(log10(x))")

        if (w==1000000):
            plt.title("WPL " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+" triples: 1M")
        else:
            plt.title("WPL " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+f" triples: {w}")

        plt.savefig("WPL(L03).png")


# plot_WPL(2)
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
    items = sorted(weights.items())

    items_1=[]
    items_2=[]

    for (k,v) in items:
        if v>4:
            items_1.append((k,v))
        else:
            items_2.append((k,v))



    x_1=[math.log(k+1,10) for (k,v) in items_1]
    y_1=[math.log(v,10) for (k,v) in items_1]
    plt.scatter( x_1 , y_1 , s=1, marker='o',color="red")

    x_2=[math.log(k+1,10) for (k,v) in items_2]
    y_2=[math.log(v,10) for (k,v) in items_2]
    plt.scatter( x_2 , y_2 , s=1, marker='o',color="blue")

    m, b = np.polyfit(x_1, y_1, 1)
    plt.plot(x_1, [(m*a + b) for a in x_1], color='red')

    plt.ylabel("count(log10(y))")
    plt.xlabel("weight(log10(x))")

    if (w==1000000):
        plt.title("SPL " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+" triples: 1M")
    else:
        plt.title("SPL " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+f" triples: {w}")

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

    items = sorted(triangles.items())

    items_1=[]
    items_2=[]

    for (k,v) in items:
        if v>4:
            items_1.append((k,v))
        else:
            items_2.append((k,v))



    x_1=[math.log(k+1,10) for (k,v) in items_1]
    y_1=[math.log(v,10) for (k,v) in items_1]
    plt.scatter( x_1 , y_1 , s=1, marker='o',color="red")

    x_2=[math.log(k+1,10) for (k,v) in items_2]
    y_2=[math.log(v,10) for (k,v) in items_2]
    plt.scatter( x_2 , y_2 , s=1, marker='o',color="blue")

    m, b = np.polyfit(x_1, y_1, 1)
    plt.plot(x_1, [(m*a + b) for a in x_1], color='red')

    plt.ylabel("count(log10(y))")
    plt.xlabel("triangles(log10(x))")

    if (w==1000000):
        plt.title("TPL " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+" triples: 1M")
    else:
        plt.title("TPL " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+f" triples: {w}")

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

    e = nx.adjacency_spectrum(G)
    e= [value.real for value in e]
    e = [-i if i < 0 else i for i in e]
    items = sorted(e)
    print(e)
    x=[math.log(x+1,10) for x in range(len(items))]
    y=[math.log(v,10) for v in items]
    plt.scatter( x , y , s=1, marker='o')
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, [(m*a + b) for a in x], color='red')

    plt.ylabel("count(log10(y))")
    plt.xlabel("triangles(log10(x))")

    if (w==1000000):
        plt.title("EPL " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+" triples: 1M")
    else:
        plt.title("EPL " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+f" triples: {w}")

    plt.savefig("EPL(L06).png")


#plot_EPL()
####################################################################################################
#End of L06: Eigenvalue Power Law (EPL)
####################################################################################################


####################################################################################################
#L07: Principal Eigenvalue Power Law (λPL)
####################################################################################################

#Plotting

def plot_λPL():

    λ_max = []
    Edges = []
    i = 1
    while (i<=w):
        G.clear() #clear all the nodes and edges
        generate_graph(G,i,elst,plst,melst,mplst,key_lst,prob)
        e = nx.adjacency_spectrum(G)
        e= [value.real for value in e]
        # e = [-i if i < 0 else i for i in e]
        λ_max.append(max(e))
        Edges.append(G.number_of_edges())
        i+=1

    x=[math.log(e,10) for e in Edges]
    y=[math.log(n,10) for n in λ_max]

    plt.scatter( x , y , s=1, marker='o')

    m, b = np.polyfit(x, y, 1)
    plt.plot(x, [(m*a + b) for a in x], color='red')


    plt.xlabel("|E|(log10(x))")
    plt.ylabel("|max(λ)|(log10(y))")

    if (w==1000000):
        plt.title("DPL " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+" triples: 1M")
    else:
        plt.title("DPL " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+f" triples: {w}")

    plt.savefig("λPL(L07).png")

# plot_λPL()

####################################################################################################
#End of L07: Principal Eigenvalue Power Law (λPL)
####################################################################################################

####################################################################################################
#L08: Small and Shrinking Diameter
####################################################################################################
# plot_L08()
def plot_L08():
    time = []
    diam = []
    i = 1
    while (i<=w):
        G.clear() #clear all the nodes and edges
        generate_graph(G,i,elst,plst,melst,mplst,key_lst,prob)
        largest = max(nx.connected_components(G), key=len)
        sub = G.subgraph(largest)
        diameter = nx.diameter(sub,e=None)
        time.append(i)
        diam.append(diameter)
        print(i)
        i+=1

    x = time
    y = diam
    plt.scatter( x , y , s=1, marker='o')

    # m, b = np.polyfit(x, y, 1)
    # plt.plot(x, [(m*a + b) for a in x], color='red')




    plt.xlabel("time")
    plt.ylabel("diameter")

    if (w==1000000):
        plt.title("L08 " +"key: "+str(key_lst)+" prob: "+str(prob)+" triples: 1M")
    else:
        plt.title("L08 " +"key: "+str(key_lst)+" prob: "+str(prob)+f" triples: {w}")

    plt.savefig("L08.png")

# plot_L08()
####################################################################################################
#End of L08: Small and Shrinking Diameter
####################################################################################################



####################################################################################################
#L09: constant size secondary and tertiary connected components
####################################################################################################
# plot_L09()
def plot_L09():
    comp1 = []
    comp2 = []
    comp3 = []
    time = []
    i = 1
    while (i<=w):
        G.clear() #clear all the nodes and edges
        generate_graph(G,i,elst,plst,melst,mplst,key_lst,prob)
        leng = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
        time.append(i)
        comp1.append(leng[0])
        if G.number_of_nodes() - leng[0] > 0:
            comp2.append(leng[1])
            if G.number_of_nodes() - leng[0] - leng[1] > 0:
                comp3.append(leng[2])
            else:
                comp3.append(0)
        else:
            comp2.append(0)
            comp3.append(0)

        i+=1

    x = time
    plt.scatter( x , comp1,s=0.5, marker='o')
    plt.scatter( x , comp2,s=0.5)
    plt.scatter( x , comp3,s=0.5)

    plt.xlabel("time")
    plt.ylabel("size of component")

    plt.savefig("L09.png")


####################################################################################################
#End of L09: constant size secondary and tertiary connected components
####################################################################################################

####################################################################################################
#L11: bursty/self-similar edge/weight additions
####################################################################################################
# plot_L11()
def plot_L11():
    Weight = []
    time = []

    current_weight = 0
    i = 1

    while (i<=w):
        G.clear() #clear all the nodes and edges
        generate_graph(G,i,elst,plst,melst,mplst,key_lst,prob)

        time.append(i)
        wt = G.size(weight='weight')
        Weight.append(wt-current_weight)
        current_weight = wt
        print(wt)
        i+=1


    x= time
    y= Weight
    plt.scatter(x , y , s=1, marker='o')

    m, b = np.polyfit(x, y, 1)
    plt.plot(x, [(m*a + b) for a in x], color='red')

    plt.ylabel("change of |W|")
    plt.xlabel("Time")

    if (w==1000000):
        plt.title("WPL(L03) " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+" triples: 1M")
    else:
        plt.title("WPL(L03) " +f"equation: {m:.3f}*x+{b:.3f}=y\n  key: "+str(key_lst)+" prob: "+str(prob)+f" triples: {w}")

    plt.savefig("L11.png")

# plot_L11()

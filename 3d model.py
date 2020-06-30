import numpy as np

keyNumber = 4 #number of keys, assume they are 'a','b','c','s'.

key_lst = ['a','b','c','s']
prob = [0.7,0.1,0.1,0.1] # Probability of each individual keys
w = 3333 # number of triples we want to generate
alpha = 0.8 #blue factor
beta = 0.9 #additional white factor
# counta = 0
# countb = 0
# countc = 0
# counts = 0
tensor = []
matrix = []
pt = []#probability tensor
pm = []#probability matrix
plst = []#probability list (tensor)
elst = []#entry list (tensor)
mplst = []#probability list (matrix)
melst = []#entry list (matrix)

# construct the tensor, append entries to corresponding list
for i in range(keyNumber):
    tensor.append([])
    pt.append([])
    for j in range(keyNumber):
        tensor[i].append([])
        pt[i].append([])
        for k in range(keyNumber):
            word = (key_lst[i]+key_lst[j]+key_lst[k])
            elst.append(word)
            Probability = prob[i]*prob[j]*prob[k]
            Probability = round(Probability,5)
            tensor[i][j].append(word)
            pt[i][j].append(Probability)
            plst.append(Probability)

# construct the matrix
for i in range(keyNumber):
    matrix.append([])
    pm.append([])
    for j in range(keyNumber):
        word = (key_lst[i]+key_lst[j])
        melst.append(word)
        Probability = round(prob[i]*prob[j],5)
        matrix[i].append(word)
        pm[i].append(Probability)
        mplst.append(Probability)

def print_tensor(ten,key):
    for i in range(key):
        for j in range(key):
            print(ten[i][j][0], ten[i][j][1], ten[i][j][2])
        print('\n')

def layerSum(ten,key,lay):
    sum = 0.0
    for i in range(key):
        for j in range(key):
            sum = sum + ten[lay][i][j]

    sum = round(sum,10)
    return sum

def rowSum(ten,key,row):
    sum = 0.0
    for i in range(key):
        for j in range(key):
            sum = sum + ten[i][row][j]
    sum = round(sum,10)
    return sum

def colSum(ten,key,col):
    sum = 0.0
    for i in range(key):
        for j in range(key):
            sum = sum + ten[i][j][col]
    sum = round(sum,10)
    return sum

def matrixRowSum(mat,key,row):
    sum = 0.0
    for i in range(key):
        sum = sum + mat[row][i]
    sum = round(sum,10)
    return sum

def imbalance(ten,key,al,be): #tensor, keyNumber, alpha, beta
    for i in range(key):
        for j in range(key):
            for k in range(key):
                if (i != j and j != k and i != k):#white
                    ten[i][j][k] = round(ten[i][j][k]*al*be,10)
                elif(i == j == k):
                    1
                else:
                    ten[i][j][k] = round(ten[i][j][k]*be,10)
    for i in range(key):
        ten[i][i][i] = ten[i][i][i]+(prob[i]-layerSum(ten,key,i))

def imbalanceMatrix(mat,key,be):#matrix, keyNumber, beta
    for i in range(key):
        for j in range(key):
            if(i != j):
                mat[i][j] = round(mat[i][j]*be,10)
    for i in range(key):
        mat[i][i] = round(mat[i][i]+(prob[i]-matrixRowSum(mat,key,i)),10)


imbalance(pt,keyNumber,alpha,beta)

imbalanceMatrix(pm,keyNumber,beta)






# print(np.sum(pt))
# print(layerSum(pt,keyNumber,2))
# print(colSum(pt,keyNumber,2))
# print(rowSum(pt,keyNumber,2))

# for i in range(keyNumber):
#     for j in range(keyNumber):
#         for k in range(keyNumber):
#             plst.append(pt[i][j][k])


def make():
    # global counta, countb, countc, counts
    L1 = ''
    L2 = ''
    L3 = ''
    T_L1 = True
    T_L2 = True
    T_L3 = True
    TNum = 3

    while(TNum == 3):
        lst = np.random.choice(elst,1,p=plst)[0]

        L1 = L1 + lst[0]

        L2 = L2 + lst[1]

        L3 = L3 + lst[2]

        if (lst[0] == 's'):
            T_L1 = False
            TNum = TNum - 1
        if (lst[1] == 's'):
            T_L2 = False
            TNum = TNum - 1
        if (lst[2] == 's'):
            T_L3 = False
            TNum = TNum - 1

    while(TNum == 2):
        lst = np.random.choice(melst,1,p=mplst)[0]
        if(T_L1 == False):
            L2 = L2 + lst[0]
            L3 = L3 + lst[1]
            if(lst[0] == 's'):
                T_L2 = False
                TNum = TNum - 1
            if(lst[1] == 's'):
                T_L3 = False
                TNum = TNum - 1
        elif(T_L2 == False):
            L1 = L1 + lst[0]
            L3 = L3 + lst[1]
            if(lst[0] == 's'):
                T_L1 = False
                TNum = TNum - 1
            if(lst[1] == 's'):
                T_L3 = False
                TNum = TNum - 1
        elif(T_L3 == False):
            L1 = L1 + lst[0]
            L2 = L2 + lst[1]
            if(lst[0] == 's'):
                T_L1 = False
                TNum = TNum - 1
            if(lst[1] == 's'):
                T_L2 = False
                TNum = TNum - 1

    while(TNum == 1):
        lst = np.random.choice(key_lst,1,p=prob)[0]
        if(T_L1 == True):
            L1 = L1 + lst[0]
        elif(T_L2 == True):
            L2 = L2 + lst[0]
        else:
            L3 = L3 + lst[0]
        if(lst[0]=='s'):
            TNum = TNum - 1



    triple = []
    triple.append(L1)
    triple.append(L2)
    triple.append(L3)


    if (not L1.endswith('s')):
        print('S1 Wrong!')
    if (not L2.endswith('s')):
        print('S2 Wrong!')
    if (not L3.endswith('s')):
        print('S3 Wrong!')
    # if L1[0]=='a':
    #     counta+=1
    # elif L1[0]=='b':
    #     countb+=1
    # elif L1[0]=='c':
    #     countc+=1
    # else:
    #     counts+=1
    #
    #
    #
    # if L2[0]=='a':
    #     counta+=1
    # elif L2[0]=='b':
    #     countb+=1
    # elif L2[0]=='c':
    #     countc+=1
    # else:
    #     counts+=1
    #
    #
    # if L3[0]=='a':
    #     counta+=1
    # elif L3[0]=='b':
    #     countb+=1
    # elif L3[0]=='c':
    #     countc+=1
    # else:
    #     counts+=1
    return triple

graph = []

for i in range(w):
    graph.append(make())

print(graph)

# print('words started with a:')
# print(counta)
# print('words started with b:')
# print(countb)
# print('words started with c:')
# print(countc)
# print('words started with s:')
# print(counts)

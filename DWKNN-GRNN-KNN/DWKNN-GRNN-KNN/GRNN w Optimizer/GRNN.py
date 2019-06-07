import math

import numpy as np
    
def GenerateDO(author, limit, dos):
    do = [0] * limit
    do[author] = dos[author]
    return do

def GenerateDO(author, limit):
    do = [0] * limit
    do[author] = 1
    return do

def getTrain():
    return train

def getNeighbors(query, test, k):
    closest = []
    for index, value in enumerate(test):
        if dist_sqrd(query[0], value[0]) != 0:
            closest.append([
                        value[0],
                        value[1],
                        dist_sqrd(query[0], value[0]),
                    ])
            
    closest = sorted(closest, key=lambda x: x[2], reverse=False)[:k]
    
    return closest

def dist_sqrd(t_q, t_i):
    sum = 0.0
    for i in range(0, len(t_q)):
        sum += pow((t_q[i] - t_i[i]),2.0)
    return sum

def hf(t_q, t_i, sigma):
    return math.exp(-dist_sqrd(t_q, t_i)/pow((2.0*sigma),2.0))


def leaveOneOut(train, sigma, k = None):
    error = 0
    for index, value in enumerate(train):
        fs_sum = [0] * len(value[1])
        test = value
        t = [x for i, x in enumerate(train) if i != index]
        dq = [0] * len(t)
        fs = [0] * len(t)
        if k == None:   
            for j, v in enumerate(t):
                fs[j] = hf(test[0],v[0],sigma)
                dq[j] = fs[j]*np.array(t[j][1])
                fs_sum += dq[j]
        else:
            k_elements = getNeighbors(test, t, k)
        
            for j, v in enumerate(k_elements):
                fs[j] = hf(test[0],v[0],sigma)
                dq[j] = fs[j]*np.array(k_elements[j][1])
                fs_sum += dq[j]
        winner = np.argmax(fs_sum)

        if winner != np.argmax(value[1]):
            error += 1

    error = error / len(train)
    return error
            
runner = 1


if runner == 0:
    fileName = 'results_tsncu.txt'
else:
    fileName = 'msst_tsncu.txt'
    
fileStream = open(fileName, 'r')
train = []
authors = []

#if runner == 2:
    #optimisation here

for line in fileStream:
   cols = line.strip().split(',')
   author = cols.pop(0)
   for index, value in enumerate(cols):
     cols[index] = float(value)
   if runner == 0:
     author = int(author.split('_')[0])-1000
   else:
     cut = author.split('_')
     author = cut[1]
     
   if author not in authors:
     authors.append(author)
   train.append([cols, author])

for index, value in enumerate(train):
     authorIndex = authors.index(value[1])
     train[index][1] = GenerateDO(authorIndex, len(authors))

        
     
sig = 0.18
#for i in range (1,6,2):
    #sig = sig + 0.01
    #print("Error for Sigma of ", sig, ":", (leaveOneOut(train,sig)*100.0))
    #print("K = 1 Error for Sigma of ", sig, ":", (leaveOneOut(train,sig, 1)*100.0))
    #print("K = 3 Error for Sigma of ", sig, ":", (leaveOneOut(train,sig, 3)*100.0))
    #print("K = 5 Error for Sigma of ", sig, ":", (leaveOneOut(train,sig, 5)*100.0))




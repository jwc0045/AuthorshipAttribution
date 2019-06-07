import math
import csv
import random
import operator
import numpy as np


def GenerateDO(author, limit):
    do = [0] * limit
    do[author] = 1
    return do

def getAuthorFromList(list):
    return np.argmax(list)

def dist_sqrd(t_q, t_i):
    sum = 0.0
    for i in range(0, len(t_q)):
        sum += math.pow((t_q[i] - t_i[i]), 2.0)
    return math.sqrt(sum)

def getNeighbors(query, tset, k):
    closest = []
    for index, value in enumerate(tset):
        if runner == 3 and dist_sqrd(value[0], query[0]) < 1.2414 or runner != 3:
            closest.append([
            dist_sqrd(value[0], query[0]),
            value[1]
        ])
        if dist_sqrd(value[0], query[0]) == 0:
            print("duplicate ??", index)
    closest = sorted(closest, key=lambda x: x[0], reverse=False)[:k]

    return closest

# train  [ Instance => [0] => feature vector, [1] => desired output ]
def leaveOneOut(train, k):
    error = 0
    for index, value in enumerate(train):
        test = value
        t = [x for i, x in enumerate(train) if i != index] # get all instances except the selected one
        if k == 7:
            k = len(t)
        neighbors = getNeighbors(test, t, k)
        sumValues = [0] * len(neighbors[0][1])
        for i in neighbors:
            sumValues += (np.array(i[1]))
        #print(value[1], "=>", neighbors, sumValues, np.argmax(sumValues))

        winningAuthor = np.argmax(sumValues)
        if winningAuthor != getAuthorFromList(value[1]):
            #print("Winner:" + str(winningAuthor) +", for:" ,index, "expected:", getAuthorFromList(value[1]), sumValues)
            #print("Winner:" + str(winningAuthor) +", for:" ,index, "expected:", getAuthorFromList(value[1]), sumValues)
            #print("for:" ,index, ":", str(winningAuthor) ,"-" ,getAuthorFromList(value[1]), neighbors)
            error += 1
    print("error rate for k=" + str(k) + " : ", 1.0*error/len(train), "success rate: ", 1 - (1.0*error/len(train)))


runner = 3


def readInstances(runner):

    if runner == 0:
        fileName = 'results_tsncu.txt'
    else:
        fileName = 'msst_tsncu.txt'

    fileStream = open(fileName, 'r')
    train = []
    authors = []

    for line in fileStream:
        cols = line.strip().split(',')
        author = cols.pop(0)
        match = -1
        for index, value in enumerate(cols):
            cols[index] = float(value)
        if runner == 0:
            author = int(author.split('_')[0]) - 1000
        else:
            #print(author)
            sepa = author.split('_')
            author = sepa[0] + sepa[1]
            match = sepa[2] + sepa[3]

        if author not in authors:
            authors.append(author)
        train.append([cols, author, match])

    for index, value in enumerate(train):
        authorIndex = authors.index(value[1])
        train[index][1] = GenerateDO(authorIndex, len(authors))
    return train, author

runner = -1
for run in range(0, 4):
    runner = run
    train, authors = readInstances(runner)

    print("Running for ", runner)
    for k in range(1, 9, 2):
        leaveOneOut(train, k)
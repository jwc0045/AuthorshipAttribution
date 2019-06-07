import random
import GRNN


def generateIndividuals(popSize, numDO):
    individuals = [0] * popSize
    
    for i in range(0, popSize):
        desired_out = [0] * numDO
        for j in range(0, numDO):
            desired_out[j] = random.gauss(0,1)
        individuals[i] = desired_out
    return individuals

def appendTrain(train, numDO, individual):
    for index, value in enumerate(train):
        for i in range(0, numDO):
            if value[1][i] != 0:
                value[1][i] = individual[i]
                #print(value[1])
    #print(value[1])
    return train

def getBestTwo(indivs):
    j = 1
    fitness = [0] * len(indivs)
    parents = [0] * 2
    for i in range(0, len(indivs)):
        #print(indivs[i])
        train = [0] * len(indivs)
        train = appendTrain(GRNN.train, 7, indivs[i])
        #print(train[i][1])
        fitness[i] = GRNN.leaveOneOut(train, 0.03)
        #print(fitness[i])
        if fitness[i] < fitness[j]:
            j = i
    parents[0] = indivs[i]
    parents[1] = indivs[j]
    #print(parents)
    return parents       
    
def generateOffspring(parents, numDO, popSize):
    delta = 0
    children = [0] * popSize
    
    for i in range(0,popSize):
        d_out = [0] * numDO
        for j in range(0, numDO):
            delta = .5 * (parents[1][j] - parents[0][j])
            dx = parents[0][j] - delta
            dy = parents[1][j] + delta
            d_out[j] = random.uniform(dx,dy)
        children[i] = d_out
        d_out = [0] * numDO
    return children

def generator(numGen, numPop, numDO, k = None):
    individuals = generateIndividuals(numPop, numDO)
    if k == None:        
        for i in range(0,numGen):
            parents = getBestTwo(individuals)
            print("Error (GEN", i, "):", (GRNN.leaveOneOut(appendTrain(GRNN.train, numDO, parents[0]), 0.03))*100)
            print("Error (GEN", i, "):", (GRNN.leaveOneOut(appendTrain(GRNN.train, numDO, parents[1]), 0.03))*100)
            children = generateOffspring(parents, numDO, (numPop-2))
            individuals = parents
            for i in range(0,len(children)):
                individuals.append(children[i])
    else:
        for i in range(0,numGen):
            parents = getBestTwo(individuals)
            print("Error (GEN", i, "):", (GRNN.leaveOneOut(appendTrain(GRNN.train, numDO, parents[0]), 0.03, k))*100)
            print("Error (GEN", i, "):", (GRNN.leaveOneOut(appendTrain(GRNN.train, numDO, parents[1]), 0.03, k))*100)
            children = generateOffspring(parents, numDO, (numPop-2))
            individuals = parents
            for i in range(0,len(children)):
                individuals.append(children[i]) 
    

#getBestTwo(generateIndividuals(20,7))
generator(100, 20, 7)




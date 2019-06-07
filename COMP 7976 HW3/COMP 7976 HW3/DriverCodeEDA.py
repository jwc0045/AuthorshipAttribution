import Data_Utils1
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import cross_val_score
import random


CU_XX, Y = Data_Utils1.Get_Casis_CUDataset("msst")

evaluated = 0
gen = 0

def evaluate(child):
    global evaluated
    evaluated += 1
    #rbfsvm = svm.SVC()
    lsvm = svm.LinearSVC()
    #mlp = MLPClassifier(max_iter=2000)
    
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    fold_accuracy = []
    CU_X = CU_XX * child
    
    scaler = StandardScaler()
    tfidf = TfidfTransformer(norm=None)
    dense = Data_Utils.DenseTransformer()
    
    for train, test in skf.split(CU_X, Y):
        #train split
        CU_train_data = CU_X[train]
        train_labels = Y[train]
        
        #test split
        CU_eval_data = CU_X[test]
        eval_labels = Y[test]
    
        # tf-idf
        tfidf.fit(CU_train_data)
        CU_train_data = dense.transform(tfidf.transform(CU_train_data))
        CU_eval_data = dense.transform(tfidf.transform(CU_eval_data))
        
        # standardization
        scaler.fit(CU_train_data)
        CU_train_data = scaler.transform(CU_train_data)
        CU_eval_data = scaler.transform(CU_eval_data)
    
        # normalization
        CU_train_data = normalize(CU_train_data)
        CU_eval_data = normalize(CU_eval_data)
    
        train_data =  CU_train_data
        eval_data = CU_eval_data
    
        # evaluation
        #rbfsvm.fit(train_data, train_labels)
        lsvm.fit(train_data, train_labels)
        #mlp.fit(train_data, train_labels)
    
        #rbfsvm_acc = rbfsvm.score(eval_data, eval_labels)
        lsvm_acc = lsvm.score(eval_data, eval_labels)
        #mlp_acc = mlp.score(eval_data, eval_labels)
    
        fold_accuracy.append((lsvm_acc))

    return np.mean(fold_accuracy, axis = 0)


def crossover(parents):
    mask = [0] * 95
    child = [0] * 2
    parentCount = len(parents)
    for i in range(0,95):
        temp = parents[np.random.randint(parentCount)][0]
        mask[i] = temp[i]
    child[0] = mask
    child[1] = -1
    return child
      
def bestFit(population):
    best = 0
    for i in range(0,len(population)):
        test = population[i][1]
        if(population[best][1] < test):
            best = i
    return best

def chooseParents(population, nParents):
    population = sorted(population, key=lambda x: float(x[1]), reverse=True)
    parents = [0] * nParents
    for i in range(nParents):
            parents[i] = population[i]
    return parents

def generateChildren(parents, popSize):
    children = [0] * popSize
    for i in range(popSize):
        children[i] = crossover(parents)
        children[i][0] = Mutation(children[i][0], 5)
        children[i][1] = evaluate(children[i][0])
    return children

def Mutation(child, rate):
    for i, _ in enumerate(child):
        if random.randint(0, 100) < rate:
            child[i] = (child[i]+1) % 2
    return child


def generatePopulation(popSize):
    pop = []
    for i in range(0,popSize):
        ind = np.random.randint(2, size=95)
        pop.append([ind, -1])
    return pop

def replaceDuplicates(generation):
    generation = sorted(generation, key=lambda x: float(x[1]), reverse=True)
    duplicates = [0] * len(generation)
    count = 0
    for i in range(len(generation)-1):
        test = generation[i+1][0]
        if (np.array_equal(generation[i][0],test)):
            duplicates[count] = (i+1)
            count = count + 1
    del duplicates[count:]
    if (count > 0):
        for i in range(len(duplicates)):
            ind = np.random.randint(2, size=95)
            generation[duplicates[i]][0] = ind
            generation[duplicates[i]][1] = evaluate(generation[duplicates[i]][0])
    return generation

def EDA(population, nParents, opt):
    global gen
    while (evaluated < 5000):
        print("Best fit on generation", gen, "| Individual", bestFit(population), "with", (population[bestFit(population)][1] * 100), "percent success rate.")
        parents = [0] * nParents
        parents = chooseParents(population, nParents)
        if (opt == True):
            children = [] * (len(population) - len(parents) + 1)
            children = generateChildren(parents, (len(population)-len(parents)))
            population = children + parents
        else:    
            children = [] * (len(population))
            children = generateChildren(parents, (len(population)-1))
            best = population[bestFit(population)]
            population = children
            population.append(best)
        gen = gen + 1
    return population   


population = generatePopulation(25)
for i, child in enumerate(population):
    population[i][1] = evaluate(child[0])
print("Moving to Estimation of Distribution Algorithm")
population = EDA(population, 12, True)
print("Best Individual in Final Generation: Individual", bestFit(population))
print(population[bestFit(population)])
popValues = np.array([indi[1] for indi in population])
avg = popValues.mean()
print("AVG: ", avg, " | BestFit: ", population[bestFit(population)][1])


    #children = generateChildren(parents)
    #for i in range(len(parents)):
     #   if (i < len(parents)):
      #      population[i] = parents[i]
       # else:
        #    population[i] = children[i-len(parents)]



    
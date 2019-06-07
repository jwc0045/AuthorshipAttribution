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

CU_XX, Y = Data_Utils1.Get_Casis_CUDataset("results")

fcount = len(CU_XX[0])
evaluated = 0

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
        # train split
        CU_train_data = CU_X[train]
        train_labels = Y[train]

        # test split
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

        train_data = CU_train_data
        eval_data = CU_eval_data

        # evaluation
        #rbfsvm.fit(train_data, train_labels)
        lsvm.fit(train_data, train_labels)
        #mlp.fit(train_data, train_labels)

        #rbfsvm_acc = rbfsvm.score(eval_data, eval_labels)
        lsvm_acc = lsvm.score(eval_data, eval_labels)
        #mlp_acc = mlp.score(eval_data, eval_labels)

        fold_accuracy.append(lsvm_acc)

    return np.mean(fold_accuracy, axis=0)

def bestfit(population):
    best = 0
    for i, gummy in enumerate(population):
        print(population[best],population[i])
        if population[best][1] < population[i][1]:
            best = i
    return best

def crossover(parents):
    parentCount = len(parents)
    child = [0] * fcount
    for i in range(fcount):
        child[i] = parents[np.random.randint(parentCount)][i]

    return child

def crossoverOpt(parents, numPar):
    child = [0] * fcount
    for i in range(fcount):
        child[i] = parents[np.random.randint(numPar)][0][i]

    return child

def generatePopulation(size):
    population = []
    for i in range(0, size):
        population.append([np.random.randint(2, size=fcount), -1])
    return population

def bestfit(population):
    best = 0
    for i, _ in enumerate(population):
        if population[best][1] < population[i][1]:
            best = i

    return best

def worstfit(population):
    worst = 0
    for i, _ in enumerate(population):
        if population[worst][1] > population[i][1]:
            worst = i

    return worst

def Mutation(child, rate):
    for i, _ in enumerate(child):
        if random.randint(0, 100) < rate:
            child[i] = (child[i]+1) % 2
            
    return child

def elitist(population, opt):
    global evaluated
    pop = population[:]
    while( evaluated < 5000 ):
        sarp = bestfit(pop)
        best = pop[sarp]
        population2 = [best]
        for i in range(0, 24):
            if opt == True:
                parents = [] * 5
                pop = sorted(pop, key=lambda x: float(x[1]), reverse=True)
                parents.append(pop[0])
                parents.append(pop[1])
                parents.append(pop[2])
                parents.append(pop[3])
                parents.append(pop[4])
                child = crossoverOpt(parents, len(parents))
                child = Mutation(child, 10)
                population2.append([child, evaluate(child)])
            else:
                Parent = np.random.randint(25, size = 2)
                parents = []
                for j in Parent:
                    parents.append(pop[j][0])
    
                child = crossover(parents)
                child = Mutation(child, 5)
                population2.append([child, evaluate(child)])

        print(evaluated)
        pop = population2

        worst = worstfit(pop)
        print( pop[worst][1], " - Best fit on", evaluated, " | ", bestfit(pop), " with ", pop[bestfit(pop)][1])
    return pop


population = generatePopulation(25)
print(population)
for i, child in enumerate(population):
    population[i][1] = evaluate(child[0])


result = elitist(population, False)
print(result)

popValues = np.array([indi[1] for indi in result])
avg = popValues.mean()

print("AVG: ", avg*100, "% | BestFit: ", result[bestfit(result)][1]*100, "%")




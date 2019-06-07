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

print(Y, CU_XX)


evaluated = 0

fcount = len(CU_XX[0])


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
        fold_accuracy.append(lsvm_acc)

    return np.mean(fold_accuracy, axis=0)


    # fold_accuracy = []
    # fold_accuracy.append((lsvm_acc, rbfsvm_acc, mlp_acc))
    #
    # print(np.mean(fold_accuracy, axis = 0))

def generatePopuplation(size):
    population = []
    for i in range(0, size):
        population.append([np.random.randint(2, size=fcount), -1])
    return population

def BestFit(population):
    best = 0
    for i, _ in enumerate(population):
        if population[best][1] < population[i][1]:
            best = i

    return best

def WorstFit(population):
    worst = 0
    for i, _ in enumerate(population):
        if population[worst][1] > population[i][1]:
            worst = i

    return worst


def Crossover(parents):
    parentCount = len(parents)
    child = [0] * fcount
    for i in range(fcount):
        child[i] = parents[np.random.random_integers(parentCount)-1][i]

    return child


def Mutation(child, rate):
    for i, _ in enumerate(child):
        if random.randint(0, 100) < rate:
            child[i] = (child[i]+1) % 2

    return child


opt = 1
tSelect = 2
pSize = 2
mRate = [5, 5]
if opt == 1:
    tSelect = 4
    pSize = 5
    mRate = [2, 6]


def SteadyState(population):
    while evaluated < 5000:
        Selected_Parents = []

        for i in range(pSize):
            parent = [population[random.randrange(len(population))] for i in range(tSelect)]
            sorted(parent, key=lambda x: float(x[1]))
            Selected_Parents.append(parent[0][0])

        child = Crossover(Selected_Parents)


        worstOne = population[WorstFit(population)][1]
        popValues = [indi[1] for indi in population]
        for i, val in enumerate(popValues):
            if(val == worstOne):
                del popValues[i]
                break

        popValues = np.array(popValues)

        avg = worstOne
        if len(popValues) > 0:
            avg = popValues.mean()

        bestWorstDifference = (population[BestFit(population)][1] - avg) * 100

        selectedRate = mRate[0]
        if mRate[0] == mRate[1]:
            selectedRate = mRate[0]
        elif bestWorstDifference < 10:
            selectedRate =  (  (mRate[1]-mRate[0])*bestWorstDifference/10  )+mRate[0]


        print("Diff: ",bestWorstDifference, ", selected rate", selectedRate)
        child = Mutation(child, selectedRate)
        worst = WorstFit(population)
        population[worst] = [child, evaluate(child)]

        print( population[worst][1], " - Best fit on", evaluated, " | ", BestFit(population), " with ", population[BestFit(population)][1])

# all_features = [1] * fcount
# gains = [0] * fcount
# all_fit = evaluate(all_features)
# mustDisabled = []
# mustEnabled = []
# for i, _ in enumerate(all_features):
#     copy = all_features[:]
#     copy[i] = 0
#     gains[i] = evaluate(copy) - all_fit
#     if(gains[i] > 0):
#         mustDisabled.append(i)
#     if(gains[i] < 0):
#         mustEnabled.append(i)
#     print(i, "-", gains[i])
#
# print(mustEnabled)
# print(mustDisabled)
#

mustEnabled = [ ]
mustDisabled = [ ]

if opt == 1:
    mustEnabled = [311, 378]
    mustDisabled = [140, 262, 273, 385]



population = generatePopuplation(25)
print(len(population))
for i, individual in enumerate(population):
    if(i < 4):
        for disable in mustDisabled:
            individual[0][i] = 0
        for disable in mustEnabled:
            individual[0][i] = 1
    population[i][1] = evaluate(individual[0])
print("Moving to steady state")
SteadyState(population)

print(population)

popValues = np.array([indi[1] for indi in population])
avg = popValues.mean()

print("AVG: ", avg, " | BestFit: ", population[BestFit(population)][1])


import Data_Utils
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
import matplotlib.pyplot as plt
CU_XX, Y = Data_Utils.Get_Casis_CUDataset("msst")
Attacks, A = Data_Utils.Get_Casis_CUDataset("amask2")

print(Y, CU_XX)
print(A, Attacks)

evaluated = 0

fcount = len(CU_XX[0])

def our_masks(mask):
    lsvm = svm.LinearSVC()
    lsvm = svm.LinearSVC()
    CU_X = CU_XX * mask
    attacks = Attacks * mask

    scaler = StandardScaler()
    tfidf = TfidfTransformer(norm=None)
    dense = Data_Utils.DenseTransformer()

    # tf-idf
    tfidf.fit(CU_X)
    CU_X = dense.transform(tfidf.transform(CU_X))
    attacks = dense.transform(tfidf.transform(attacks))

    # standardization
    scaler.fit(CU_X)
    CU_X = scaler.transform(CU_X)
    attacks = scaler.transform(attacks)

    # normalization
    CU_X = normalize(CU_X)
    attacks = normalize(attacks)

    for ai, attack in enumerate(attacks):

        index = list(Y).index(A[ai])

        if index > 0:
            train = list(CU_X[:index]) + list(CU_X[index+1:])
            train_labels  = list(Y[:index]) + list(Y[index+1:])
        else:
            train =  list(CU_X[index+1:])
            train_labels  =  list(Y[index+1:])



        # rbfsvm.fit(train_data, train_labels)
        lsvm.fit(train, [line.split('_')[1] for line in train_labels])
        # mlp.fit(train_data, train_labels)

        # rbfsvm_acc = rbfsvm.score(eval_data, eval_labels)
        prediction = lsvm.predict([attack])

        res =  np.amax(lsvm.decision_function([attack]))
        if res < -0.1:
            print("Modified ignored --" + A[ai] +" - Classified as: ", prediction[0], "-- ", res)
        else:
            print("Valid sample -- " + A[ai] +" - Classified as: ", prediction[0])



def attacks(vectors, mask):
    lsvm = svm.LinearSVC()
    CU_X = CU_XX * mask
    attack = vectors * mask

    scaler = StandardScaler()
    tfidf = TfidfTransformer(norm=None)
    dense = Data_Utils.DenseTransformer()

    # tf-idf
    tfidf.fit(CU_X)
    CU_X = dense.transform(tfidf.transform(CU_X))
    attack = dense.transform(tfidf.transform(attack))

    # standardization
    scaler.fit(CU_X)
    CU_X = scaler.transform(CU_X)
    attack = scaler.transform(attack)

    # normalization
    CU_X = normalize(CU_X)
    attack = normalize(attack)


    # evaluation
    # rbfsvm.fit(train_data, train_labels)
    lsvm.fit(CU_X, Y)
    # mlp.fit(train_data, train_labels)

    # rbfsvm_acc = rbfsvm.score(eval_data, eval_labels)
    prediction = lsvm.predict(attack)

    results = [ np.amax(a) for a in lsvm.decision_function(attack) ]
    for i, res in enumerate(results):
        if res < -0.1:
            print("Modified ignored --" + A[i] + " - ", res)
        else:
            print("Valid sample -- " + A[i] +" - Classified as: ", prediction[i])


    return prediction

def evaluate(child):
    global evaluated
    evaluated += 1
    #rbfsvm = svm.SVC()
    lsvm = svm.LinearSVC()
    #mlp = MLPClassifier(max_iter=2000)

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
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


resm = [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0]

print(our_masks(resm))

#print(attacks(Attacks, resm))

# MSST      - 96.875
# [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0]

# results   - 81.0
# [1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1]
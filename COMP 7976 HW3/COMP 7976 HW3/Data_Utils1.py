import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def Get_Casis_CUDataset(filename):
    X = []
    Y = []
    with open(filename+"_rawcu.txt", "r") as feature_file:
        for line in feature_file:
            line = line.strip().split(",")
            #print(line[0])
            if filename == "results":
                Y.append(line[0].split('_')[0])
            else:
                sepa = line[0].split('_')
                Y.append(line[0].split('_')[1])

            X.append([float(x) for x in line[1:]])
    return np.array(X), np.array(Y)


class DenseTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray()



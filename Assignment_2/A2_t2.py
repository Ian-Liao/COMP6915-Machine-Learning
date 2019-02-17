import numpy as np
from numpy import *
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot

import sys
import csv


class KFoldCrossValidation(object):

    def __init__(self, k_fold, k_min, k_max, k_step):
        self.k_fold = k_fold
        self.k_min = k_min
        self.k_max = k_max
        self.k_step = k_step
        self.offset = 1

    def split(self, array, k, index, class_boundary):
        """ split class 0 into k folds and class 1 into k folds separately and then combine
         them to training set and testing set
        """

        class1_set, class0_set = array[:class_boundary], array[class_boundary:]
        size1 = class1_set.shape[0]
        start1 = (size1//k) * index
        end1 = (size1//k) * (index+1)
        testing1 = class1_set[start1:end1]
        training1 = np.concatenate((class1_set[:start1], class1_set[end1:]))

        size0 = class0_set.shape[0]
        start0 = (size0//k) * index
        end0 = (size0//k) * (index+1)
        testing0 = class0_set[start0:end0]
        training0 = np.concatenate((class0_set[:start0], class0_set[end0:]))

        training = np.concatenate((training0, training1))
        testing = np.concatenate((testing0, testing1))
        return training, testing

    def KFoldCrossValidation(self, learner, features, targets):
        col_num = features.shape[1]
        # score_matrix stores scores for different KNN and features selection
        score_matrix = pd.DataFrame(np.zeros((col_num, self.k_max)))

        train_folds_score = []
        validation_folds_score = []
        for index in range(self.k_fold):
            unique, counts = np.unique(targets, return_counts=True)
            counter = dict(zip(unique, counts))
            # print('counter: ', counter)
            y_train, y_test = self.split(targets, self.k_fold, index, counter[1])
            X_train, X_test = self.split(features, self.k_fold, index, counter[1])
            for cn in range(self.offset, col_num+self.offset):
                for k in range(self.k_min, self.k_max, self.k_step):
                    knn = learner(n_neighbors = k)
                    # fit a model
                    training_predicted = knn.fit(X_train[:, :cn], y_train)
                    validation_predicted = knn.predict(X_test[:, :cn])
                    accuracy = knn.score(X_test, y_test)
                    # store the cummulativescore of KNN for each fold
                    score_matrix.iloc[cn, k] += accuracy

        score_matrix = score_matrix / self.k_fold
        print(score_matrix)
        max_score = 0.0
        best_feature_num, best_k_num = 0, 0
        for i in range(col_num):
            for j in range(self.k_max - 1):
                if (score_matrix.iloc[i, j] > max_score):
                    best_feature_num = i + 1
                    best_k_num = j + 1
                    max_score = score_matrix.iloc[i, j]

        print(
            "The best number of features is {}, The best number of neighbors is {}".format(
                best_feature_num,
                best_k_num))
        print(
            "The accuracy for aforementioned values is: {0:.4f}".format(max_score))
        return best_feature_num, best_k_num


def prepare_data(train_data):
    train_data = array(train_data)
    # attributes, labels = hsplit(train_data, [-2])
    train_data = train_data.astype(float)
    train_data = pd.DataFrame(train_data) # convert data type from numpy.array to pandas.DataFrame
    print(train_data.shape)

    # split features and targets
    features = train_data.iloc[:, :-1]
    targets = train_data.iloc[:, -1:]
    # Sort features by variance in descending order
    features = features.reindex(features.var().sort_values(ascending=False).index, axis=1)
    # Create VarianceThreshold object with a variance with a threshold of 0.1
    thresholder = VarianceThreshold(threshold=.1)

    # Conduct variance thresholding
    features = thresholder.fit_transform(features)

    # TODO consider add correlation filter
    print("***columns: ", features.shape[1])
    # Convert targets from DataFrames to numpy.array and the value type from float to int
    return features, targets.values.astype(int)

if __name__ == "__main__":
    from ipdb import set_trace
    observation_file = sys.argv[1]
    observations = []
    with open(observation_file) as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        for row in reader:
            observations.append(row)

    features, targets = prepare_data(observations)

    k_fold = 5
    k_min = 3
    k_max = 17
    k_step = 2
    kfcv = KFoldCrossValidation(k_fold=k_fold, k_min=k_min, k_max=k_max, k_step=k_step)
    kfcv.KFoldCrossValidation(KNeighborsClassifier, features, targets)


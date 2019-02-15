import numpy as np
from numpy import *
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold

import sys
import csv


class KFoldCrossValidation(object):

    def __init__(self, k_fold, k_max):
        self.k_fold = k_fold
        self.k_max = k_max

    def split(self, array, k, index):
        # TODO split class 0 into k folds and class 1 into k folds separately and then combine
        # them to training set and testing set
        size = array.shape[0]
        start = (size//k) * index
        end = (size//k) * (index+1)
        testing = array[start:end]
        training = np.concatenate((array[:start], array[end:]))
        return training, testing

    def KFoldCrossValidation(self, learner, features, targets):
        train_folds_score = []
        validation_folds_score = []
        for index in range(self.k_fold):
            training_set, testing_set = self.split(features, self.k_fold, index)
            training_targets, testing_targets = self.split(targets, self.k_fold, index)
            col_num = features.shape[1]
            for cn in range(1, col_num+1):
                for k in range(3, self.k_max, 2):
                    knn = learner(n_neighbors = k)
                    training_predicted = knn.fit(training_set[:, :cn], training_targets)
                    validation_predicted = knn.predict(testing_set[:, :cn])
                    train_folds_score.append(metrics.accuracy_score(training_targets, training_predicted))
                    validation_folds_score.append(metrics.accuracy_score(validation_targets, validation_predicted))
        return train_folds_score, validation_folds_score


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
    # filter features by the variance and correlation
    return features, targets.values

if __name__ == "__main__":
    from ipdb import set_trace
    observation_file = sys.argv[1]
    observations = []
    with open(observation_file) as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        for row in reader:
            observations.append(row)

    features, targets = prepare_data(observations)

    # count the number of class 0 and class 1
    unique, counts = unique(targets, return_counts=True)
    counter = dict(zip(unique, counts))
    print('counter: ', counter)
    k_fold = 5
    k_max = 17
    kfcv = KFoldCrossValidation(k_fold=k_fold, k_max=k_max)
    kfcv.KFoldCrossValidation(KNeighborsClassifier, features, targets)


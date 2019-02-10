import numpy as np
from numpy import *
import pandas as pd

import sys
import csv


class KFoldCrossValidation(object):

    def split(self, vector, k, index):
        size = vector.shape[0]
        start = (size/k) * index
        end = (size/k) * (index+1)
        validation = vector[start:end]
        if str(type(vector)) == "<class 'scipy.sparse.csr.csr_matrix'>":
            indices = range(start, end)
            mask = np.ones(vector.shape[0], dtype=bool)
            mask[indices] = False
            training = vector[mask]
        elif str(type(vector)) == "<type 'numpy.ndarray'>":
            training = np.concatenate((vector[:start], vector[end:]))
        return training, testing

    def KFoldCrossValidation(self, learner, k, examples, labels):
        train_folds_score = []
        validation_folds_score = []
        for index in range(k):
            training_set, testing_set = self.split(examples, k, index)
            training_labels, testing_labels = self.split(labels, k, index)
            learner.fit(training_set, training_labels)
            training_predicted = learner.predict(training_set)
            validation_predicted = learner.predict(validation_set)
            train_folds_score.append(metrics.accuracy_score(training_labels, training_predicted))
            validation_folds_score.append(metrics.accuracy_score(validation_labels, validation_predicted))
        return train_folds_score, validation_folds_score


def prepare_data(train_data):
    train_data = array(train_data)
    # attributes, labels = hsplit(train_data, [-1])
    train_data = train_data.astype(float)
    train_data = pd.DataFrame(train_data)
    print(train_data.shape)
    var_list, cov_list = [], []
    feature_num = train_data.shape[1] - 1
    for i in range(feature_num):
        corr = train_data[[i, feature_num]].corr()
        var_list.append((i, train_data[i].var(), corr[feature_num][i]))
    var_list = [var for var in var_list if var[1] > 1e-4 and abs(var[2]) >= 1e-2]
    index_set = {var[0] for var in var_list}
    print("var_list length: ", len(var_list))
    print(index_set)
    features = train_data
    return train_data

if __name__ == "__main__":
    observation_file = sys.argv[1]
    observations = []
    with open(observation_file) as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        for row in reader:
            observations.append(row)

    train_data = prepare_data(observations)

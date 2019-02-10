import numpy as np
from numpy import *
import pandas as pd

import sys
import csv


def KFoldCrossValidation(k=5):
    pass


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

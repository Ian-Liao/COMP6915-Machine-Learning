import numpy as np
from numpy import *

import sys
import csv

def calculate_metrics(labels, data, total_predicted, total_observed):
    """calculate over accuracy, precision, recall, specificity and FDR"""
    _sum = 0
    precision, recall, specificity, FDR = [], [], [], []
    for i in range(len(labels)):
        _sum += data[i][i]
        pre = data[i][i] / total_predicted[i]
        precision.append(pre)
        obs = data[i][i] / total_observed[i]
        recall.append(obs)
    accuracy = _sum / total_predicted.sum()

    return precision, recall, specificity, FDR, accuracy

def prepare_data(matrix_data):
    """reorganize the data in confusion matrix"""
    data_rows = array(matrix_data[1:])
    labels, data = hsplit(data_rows, [1])
    data = data.astype(int)
    total_predicted = data.sum(axis=1)
    total_observed = data.sum(axis=0)

    return labels, data, total_predicted, total_observed

if __name__ == "__main__":
    confusion_matrix = sys.argv[1]
    matrix_data = []
    with open(confusion_matrix) as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        for row in reader:
            matrix_data.append(row)

    labels, data, total_predicted, total_observed = prepare_data(matrix_data)
    calculate_metrics(labels, data, total_predicted, total_observed)
    print(labels)
    print(data)
    print(data[0][0])
    print(data[1][1])
    print(total_predicted)
    print(total_observed)


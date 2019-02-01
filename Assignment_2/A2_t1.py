import numpy as np
from numpy import *

import sys
import csv

def calculate_metrics(labels, data, total_predicted, total_observed):
    """calculate over accuracy, precision, recall, specificity and FDR"""
    _sum = 0
    all_sum = total_predicted.sum() # the sum of all values in the confusion matrix
    precision, recall, specificity, FDR = [], [], [], []
    for i in range(len(labels)):
        _sum += data[i][i]
        # calculate the precision of each class
        pre = data[i][i] / total_predicted[i]
        precision.append(pre)
        # calculate the recall of each class
        obs = data[i][i] / total_observed[i]
        recall.append(obs)
        # calculate the specificity of each class
        numerator = all_sum - total_observed[i] - total_predicted[i] + data[i][i]
        denominator = all_sum - total_observed[i]
        spe = numerator / denominator
        specificity.append(spe)
        # calculate the FDR of each class
        fdr = (total_predicted[i] - data[i][i]) / total_predicted[i]
        FDR.append(fdr)
    # calculate the over all accuracy
    accuracy = _sum / all_sum

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
    # from ipdb import set_trace
    # set_trace()
    confusion_matrix = sys.argv[1]
    matrix_data = []
    with open(confusion_matrix) as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        for row in reader:
            matrix_data.append(row)

    labels, data, total_predicted, total_observed = prepare_data(matrix_data)
    precision, recall, specificity, FDR, accuracy = calculate_metrics(labels, data, total_predicted, total_observed)
    print(precision)
    print(recall)
    print(specificity)
    print(FDR)
    print(accuracy)


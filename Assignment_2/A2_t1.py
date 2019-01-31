import numpy as np
from numpy import *

import sys
import csv

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
    print(labels)
    print(data)
    print(total_predicted)
    print(total_observed)


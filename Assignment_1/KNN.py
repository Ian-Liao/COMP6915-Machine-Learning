import numpy as np
from numpy import *

import sys
import operator
import csv


def knn_classifier(test, train_set, labels, k=3):
    ''' Construct KNN classifier
    test: input vector, data to be tested
    train_set: training data extracted from train.tsv
    labels: labels for the training data
    k: k value in KNN, the default value is 3
    '''

    distances = EuclideanDistance(test, train_set)
    # print('the distance between vector test and each of train_set:\n'+str(distances))

    sortedDistIndexs = distances.argsort(axis=0)  # sort the distances in ascending order
    # print(sortedDistIndicies)

    classCount = {}   # get the classes of first k nodes
    for i in range(k):
        votelabel = labels[sortedDistIndexs[i]]
        classCount[votelabel] = classCount.get(votelabel, 0) + 1 # count appearance based on class
    # get the most likely class, itemgetter(0) means sorted by key, itemgetter(1) means sorted by value
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # print(str(vecX)+'KNN decision for this unseen instance：\n'+str(sortedClassCount[0][0]))
    return sortedClassCount[0][0]

def EuclideanDistance(test, train_set):
    # Step 1.get row and col of train_set
    row, col = train_set.shape
    # Step 2.calculate difference between test data and each vector in train_set
    # tile function repeats vector test 'row' times on vertical direction and 1 time on horizontal direstion
    print(tile(test, (row, 1)))
    differences = tile(test, (row, 1)) - train_set
    # Step 3.calculate power of difference
    diff_power = differences ** 2
    # Step 4.calculate Euclidean Distance
    euc_dis = diff_power.sum(axis=1) ** 0.5 # axis=0 means sum on column, axis=1 means sum on row
    return euc_dis

def prepare_data(train_data):
    train_data = array(train_data[1:])
    attributes, labels = hsplit(train_data, [9])
    attributes = attributes.astype(float)
    labels = [label[0] for label in labels]
    return attributes, labels


if __name__ == "__main__":
    #from ipdb import set_trace
    print('sys.argv is ', sys.argv)
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    k  = int(sys.argv[3])
    train_data = []
    with open(train_file) as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        for row in reader:
            train_data.append(row)
    test_data = []
    with open(test_file) as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        for row in reader:
            test_data.append(row)

    attributes, labels = prepare_data(train_data)
    print('attributes: ', attributes, ' labels: ', labels)
    result = []
    test_data = [[float(d) for d in data] for data in test_data[1:]]
    for data in test_data:
        res = knn_classifier(data, attributes, labels, k)
        result.append(res)

    print(result)


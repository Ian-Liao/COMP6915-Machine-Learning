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
    k:k value in KNN, the default value is 3
    '''
    # TODO: remove these lines below
    dataSetSize = dataset.shape[0]
    # tile方法是在列向量vecX，datasetSize次，行向量vecX1次叠加
    diffMat = tile(vecX,(dataSetSize,1)) - dataset
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)   # axis=0 是列相加,axis=1是行相加
    distances = sqDistances**0.5

    distances = EuclideanDistance(test, train_set)
    # print('vecX向量到数据集各点距离：\n'+str(distances))

    sortedDistIndexs = distances.argsort(axis=0)  # sort the distances in ascending order
    # print(sortedDistIndicies)

    classCount = {}   # 统计前k个类别出现频率
    for i in range(k):
        votelabel = labels[sortedDistIndexs[i]]
        classCount[votelabel] = classCount.get(votelabel,0) + 1 #统计键值
    # 类别频率出现最高的点,itemgetter(0)按照key排序，itemgetter(1)按照value排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    # print(str(vecX)+'KNN的投票决策结果：\n'+str(sortedClassCount[0][0]))
    return sortedClassCount[0][0]

def EuclideanDistance(test, train_set):
    # Step 1.get row and col of train_set
    row, col = train_set.shape
    # Step 2.calculate difference between test data and each vector in train_set
    differences = tile(test, (row, 1)) - train_set
    # Step 3.calculate power of difference
    diff_power = differences ** 2
    # Step 4.calculate Euclidean Distance
    euc_dis = diff_power.sum(axis=1) ** 0.5
    return euc_dis

def prepare_data(train_data):
    train_data = array(train_data[1:])
    attributes, labels = hsplit(train_data, [9])
    labels = [label[0] for label in labels]
    return attributes, labels


if __name__ == "__main__":
    # from ipdb import set_trace
    # set_trace()
    print('sys.argv is ', sys.argv)
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    k  = sys.argv[3]
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
    for data in test_data[1:]:
        res = knn_classifier(data, attributes, labels, k)
        result.append(res)

    print(result)


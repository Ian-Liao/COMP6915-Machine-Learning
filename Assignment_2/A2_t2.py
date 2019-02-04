import numpy as np
from numpy import *

import sys
import csv


def KFoldCrossValidation(k=5):
    pass


def prepare_data(train_data):
    train_data = array(train_data[1:])
    attributes, labels = hsplit(train_data, [-1])
    attributes = attributes.astype(float)
    labels = [label[0] for label in labels]
    return attributes, labels

if __name__ == "__main__":
    observation_file = sys.argv[1]
    observations = []
    with open(observation_file) as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        for row in reader:
            observations.append(row)

    attributes, labels = prepare_data(observations)
    print(attributes.shape)
    print(labels)

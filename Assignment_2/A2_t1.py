import numpy as np
from numpy import *

import sys
import csv


if __name__ == "__main__":
    confusion_matrix = sys.argv[1]
    matrix_data = []
    with open(confusion_matrix) as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        for row in reader:
            matrix_data.append(row)

    print(matrix_data)


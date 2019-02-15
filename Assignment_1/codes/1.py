# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:16:08 2019

@author: Javad
"""

import pandas as np
from pandas import read_csv
D= read_csv("a.tsv",sep="\t",header=None)
d=D.values
print(d)
##################3

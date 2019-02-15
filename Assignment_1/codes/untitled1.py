# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:25:47 2019

@author: Javad
"""

import numpy as np
import pandas as pd

c=pd.Series(["C1","C2","C3"])
print(c)
ff=pd.DataFrame(np.arange(3,15).reshape(3,4))
ff=ff.set_index(c)
print(ff)

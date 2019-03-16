import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# Reading the data
ftrain = open("TrainingData_A1.tsv")
ftrain.readline()  # skip the header
data = np.loadtxt(ftrain)
Xtrain = data[:, 0:data.shape[1]-1]  
ytrain = data[:, data.shape[1]-1] 
ftest= open("TestData_A1.tsv")
ftest.readline()
Xtest= np.loadtxt(ftest)


#applying KNN
knn= KNeighborsClassifier(n_neighbors=3,metric='euclidean') # change k here
knn.fit(Xtrain,ytrain)
yTest=knn.predict(Xtest)
prob= knn.predict_proba(Xtest)
for i in range(10):
    print(yTest[i],"  {0:0.2f}".format(max(prob[i,:])))

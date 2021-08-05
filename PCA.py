import os, glob
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

path = "C:/Users/David Bakker/Documents/GitHub/Data sets/iris.data"
with open(path, 'r') as file:
    lines = file.readlines()

data = []
labels = {'Iris-virginica': 0, 'Iris-setosa': 1, 'Iris-versicolor': 2}
indices = []

for line in lines:
    line = line[:-1].split(",")
    
    if(len(line) != 1):
        data.append([float(line[i]) for i in range(len(line) - 1)])

        if(line[-1] == 'Iris-virginica'):
            indices.append(labels['Iris-virginica'])
        if(line[-1] == 'Iris-setosa'):
            indices.append(labels['Iris-setosa'])
        if(line[-1] == 'Iris-versicolor'):
            indices.append(labels['Iris-versicolor'])

data = np.array(data)

X = np.zeros([len(data), 2])
X[:, 0] = data[:, 2]
X[:, 1] = data[:, 3]

#print(X)
#print(indices)

def PCA(X, num_comp):
    X[:, 0] = X[:, 0] - np.mean(X[:, 0])
    X[:, 1] = X[:, 1] - np.mean(X[:, 1])

    C = np.matmul(np.transpose(X), X)
    L, W = LA.eig(C)

    idx = np.argsort(L[::-1])
    L = L[idx]
    W = W[:, idx]

    W_trans = W[:, :num_comp]

    return np.matmul(X, W_trans)

X_trans = PCA(X, 2)
print(X)
fig, ax = plt.subplots(1, 2, figsize = (10, 5))
for i in range(len(X_trans)):
    colors = ['r', 'g', 'b']
    idx = indices[i]
    ax[1].scatter(X_trans[i, 0], X_trans[i, 1], color = colors[idx])
    ax[0].scatter(X[i, 0], X[i,1], color = colors[idx])
plt.show()
#plt.scatter(setosa[:, 2], setosa[:, 3], color = 'r')
#plt.scatter(versicolor[:, 2], versicolor[:, 3], color ='g')
#plt.scatter(virginica[:, 2], virginica[:, 3], color = 'b')
#plt.show()
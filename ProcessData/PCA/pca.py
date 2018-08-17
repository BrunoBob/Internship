#import pandas as pd
import csv
from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA

FREQ = 12
FPS = 12
TIME = 1
MARKER = 16
COMP = 50

def printDataChunk(data, labels):
    X = np.zeros((MARKER,TIME*FREQ))
    Y = np.zeros((MARKER,TIME*FREQ))
    color = ['bo', 'go','ro', 'co','mo', 'yo','ko', 'bv','gv','rv', 'cv','mv', 'yv','kv','gs','rs', 'cs']
    for i in range(0,MARKER):
        for j in range(TIME*FREQ):
            X[i][j] = data[MARKER*j*3 + i*3]
            Y[i][j] = data[MARKER*j*3 + i*3 +1]

    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(10, 10))
        for marker in range(0,MARKER):
            plt.plot(X[marker,:],Y[marker,:], color[marker], label = labels[marker*3])
        plt.ylabel('Y position')
        plt.xlabel('X position')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

#read the data
f = open('../../Data/Bruno_1_juin/processedMotion.csv', 'rb')
num_lines = sum(1 for line in open('../../Data/Bruno_1_juin/processedMotion.csv'))

#put data in matrix
reader = csv.reader(f)
labels = next(reader)
next(reader)
print(num_lines)
X_temp = np.zeros((num_lines,MARKER*3))
for i in range(2,num_lines):
    X_temp[i,:] = next(reader)[0:MARKER*3]

X = X_temp[2:,:]

#Put data in the right format (sequence of movement)
num = (X.shape[0]/FPS) * TIME
size = MARKER*3*FREQ
X_std = np.zeros((num+1,MARKER*3*FREQ))
print(X_std.shape)

iter = 0
for i in range(0,num-1):
    for j in range(0,FREQ):
       	X_std[i][j*MARKER*3:(j+1)*MARKER*3] = X[iter]
        iter += FPS/FREQ
	#print(iter)

printDataChunk(X_std[100], labels)

sklearn_pca = sklearnPCA(n_components=COMP) 
sklearn_pca =sklearn_pca.fit(X_std)

#print(sklearn_pca.explained_variance_ratio_)
Y = sklearn_pca.transform(X_std)
#print(Y[0])

Xnew =  sklearn_pca.inverse_transform(Y)
printDataChunk(Xnew[100], labels)

Y_test = np.random.rand(2,50) 
X_test =  sklearn_pca.inverse_transform(Y_test)
#printDataChunk(X_test[0], labels)

cum_var_exp = np.cumsum(sklearn_pca.explained_variance_ratio_)
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 10))
    plt.bar(range(COMP), sklearn_pca.explained_variance_ratio_, alpha=0.5, align='center',label='individual explained variance')
    plt.step(range(COMP), cum_var_exp, where='mid',label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    

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

def printDataChunk(data, labels):
    X = np.zeros((MARKER,TIME*FREQ))
    Y = np.zeros((MARKER,TIME*FREQ))
    color = ['bo', 'go','ro', 'co','mo', 'yo','ko', 'bv','gv','rv', 'cv','mv', 'yv','kv','gs','rs', 'cs']
    for i in range(0,MARKER):
        for j in range(TIME*FREQ):
            X[i][j] = data[MARKER*j*3 + i*3]
            Y[i][j] = data[MARKER*j*3 + i*3 +1]

    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6, 6))
        for marker in range(0,MARKER):
            plt.plot(X[marker,:],Y[marker,:], color[marker], label = labels[marker*3])
        plt.ylabel('Y position')
        plt.xlabel('X position')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

""" df = pd.read_csv(
    #filepath_or_buffer='../../Data/iris/iris.csv',
    filepath_or_buffer='../../Data/Bruno_1_juin/processedMotion.csv',
    header=None,
    sep=',') """


f = open('../../Data/Bruno_1_juin/processedMotion.csv', 'rb')
num_lines = sum(1 for line in open('../../Data/Bruno_1_juin/processedMotion.csv'))

reader = csv.reader(f)
labels = next(reader)
next(reader)
print(num_lines)
X = np.zeros((num_lines,MARKER*3))
for i in range(2,num_lines):
    X[i,:] = next(reader)

#X = df.ix[2:,:].values


num = (X.shape[0]/FPS) * TIME
size = MARKER*3*FREQ
X_time = np.zeros((num+1,MARKER*3*FREQ))
print(X_time.shape)
iter = 0
for i in range(0,num-1):
    for j in range(0,FREQ-1):
       	X_time[i][j*MARKER*3:(j+1)*MARKER*3] = X[iter]
        iter += FPS/FREQ
	#print(iter)

printDataChunk(X_time[0], labels)

X_std = StandardScaler().fit_transform(X_time)
#print(X_time)
sklearn_pca = sklearnPCA() #n_components=2)
sklearn_pca.fit(X_std)
#print(sklearn_pca.get_covariance())
#print(sklearn_pca.explained_variance_ratio_)
Y = sklearn_pca.transform(X_std)
#print(Y)
Xnew =  sklearn_pca.inverse_transform(Y)
printDataChunk(Xnew[0], labels)
"""cum_var_exp = np.cumsum(sklearn_pca.explained_variance_ratio_)
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, size))
    plt.bar(range(size), sklearn_pca.explained_variance_ratio_, alpha=0.5, align='center',label='individual explained variance')
    plt.step(range(size), cum_var_exp, where='mid',label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    #plt.show()
    fig.savefig("test")"""

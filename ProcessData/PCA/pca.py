#import pandas as pd
import csv
from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA

#Perform a PCA to reduce the number of dimension and generate motion file with random noise
#Input : A csv processed file obtained with plotFace.py

FREQ = 30 #number of iteration for 1 second
FPS = 60 #fps of input data
TIME = 2 #number of seconds of motion
MARKER = 16 #number of marker
COMP = 7 #number of principal components kept

REF_CENTER = 0
REF_LEFT = 0
REF_UP = 0
FOREHEAD_RIGHT = 0
FOREHEAD_LEFT = 0
FOREHEAD = 0
CHEEK_RIGHT = 0
CHEEK_LEFT = 0
EYE_RIGHT = 0
EYE_LEFT = 0
EYEBROW_RIGHT = 0
EYEBROW_LEFT = 0
NOSE_RIGHT = 0
NOSE_LEFT = 0
MOUTH_RIGHT = 0
MOUTH_LEFT = 0
MOUTH_UP = 0
MOUTH_DOWN = 0
CHIN = 0

#print the graph for each frame of a given sequence
def printDataChunk(data, labels,name):
    X = np.zeros((MARKER,TIME*FREQ))
    Y = np.zeros((MARKER,TIME*FREQ))
    color = ['bo', 'go','ro', 'co','mo', 'yo','ko', 'bv','gv','rv', 'cv','mv', 'yv','kv','gs','rs', 'cs']
    for i in range(0,MARKER):
        for j in range(TIME*FREQ):
            X[i][j] = data[MARKER*j*3 + i*3]
            Y[i][j] = data[MARKER*j*3 + i*3 +1]

    with plt.style.context('seaborn-whitegrid'):
        for iter in range(0,TIME*FREQ):
            plt.figure(figsize=(10, 10))
            plt.axis([0.2, 0.6, 0, 0.6])
            for marker in range(0,MARKER):
				plt.plot(X[marker,iter],Y[marker,iter], color[marker], label = labels[marker*3])

            #Draw the mouth
            plt.plot([X[MOUTH_UP][iter], X[MOUTH_LEFT][iter]], [Y[MOUTH_UP][iter], Y[MOUTH_LEFT][iter]],color='r', linewidth=2.0)
            plt.plot([X[MOUTH_UP][iter], X[MOUTH_RIGHT][iter]], [Y[MOUTH_UP][iter], Y[MOUTH_RIGHT][iter]],color='r', linewidth=2.0)
            plt.plot([X[MOUTH_DOWN][iter], X[MOUTH_LEFT][iter]], [Y[MOUTH_DOWN][iter], Y[MOUTH_LEFT][iter]],color='r', linewidth=2.0)
            plt.plot([X[MOUTH_DOWN][iter], X[MOUTH_RIGHT][iter]], [Y[MOUTH_DOWN][iter], Y[MOUTH_RIGHT][iter]],color='r', linewidth=2.0)
            
            #draw the eyes
            l1 = (Y[EYEBROW_RIGHT][iter] - Y[EYE_RIGHT][iter]) /2
            plt.plot([X[EYE_RIGHT][iter], X[EYE_RIGHT][iter]+l1], [Y[EYE_RIGHT][iter], Y[EYEBROW_RIGHT][iter]-3*l1/2],color='r', linewidth=2.0)
            plt.plot([X[EYE_RIGHT][iter], X[EYE_RIGHT][iter]-l1], [Y[EYE_RIGHT][iter], Y[EYEBROW_RIGHT][iter]-3*l1/2],color='r', linewidth=2.0)
            plt.plot([X[EYEBROW_RIGHT][iter], X[EYE_RIGHT][iter]+l1], [Y[EYEBROW_RIGHT][iter]-l1, Y[EYEBROW_RIGHT][iter]-3*l1/2],color='r', linewidth=2.0)
            plt.plot([X[EYEBROW_RIGHT][iter], X[EYE_RIGHT][iter]-l1], [Y[EYEBROW_RIGHT][iter]-l1, Y[EYEBROW_RIGHT][iter]-3*l1/2],color='r', linewidth=2.0)
            plt.plot([X[EYE_LEFT][iter], X[EYE_LEFT][iter]+l1], [Y[EYE_LEFT][iter], Y[EYEBROW_LEFT][iter]-3*l1/2],color='r', linewidth=2.0)
            plt.plot([X[EYE_LEFT][iter], X[EYE_LEFT][iter]-l1], [Y[EYE_LEFT][iter], Y[EYEBROW_LEFT][iter]-3*l1/2],color='r', linewidth=2.0)
            plt.plot([X[EYEBROW_LEFT][iter], X[EYE_LEFT][iter]+l1], [Y[EYEBROW_LEFT][iter]-l1, Y[EYEBROW_LEFT][iter]-3*l1/2],color='r', linewidth=2.0)
            plt.plot([X[EYEBROW_LEFT][iter], X[EYE_LEFT][iter]-l1], [Y[EYEBROW_LEFT][iter]-l1, Y[EYEBROW_LEFT][iter]-3*l1/2],color='r', linewidth=2.0)
            
            plt.ylabel('Y position')
            plt.xlabel('X position')
            plt.legend(loc='best')
            plt.tight_layout()

            plt.savefig("graph/graph"+name + str(iter))
            plt.close()

#save a sequence in a csv file
#Format : time marker1X marker1Y marker1Z marker2X ...
def saveData(labels, data, name):
    f = open(name, 'wb')
    writer = csv.writer(f)
    label = ["" for x in range(MARKER*3+1)]
    label[0] = "time_seconds"
    label[1:MARKER*3] = labels
    writer.writerow(label)
    current_data = np.zeros(MARKER*3 + 1)
    for line in range(0,TIME * FREQ):
        current_data[0] = float(line)/float(FREQ)
        current_data[1:MARKER*3+1] = data[line * MARKER * 3 : (line+1) * MARKER * 3]
        writer.writerow(current_data)


#read the data
f = open('../../Data/Jiro/processedMotion.csv', 'rb')
num_lines = sum(1 for line in open('../../Data/Jiro/processedMotion.csv'))

#put data in matrix
reader = csv.reader(f)
labels = next(reader)

#check presence of each marker
for marker in range(0, MARKER):
	if(labels[marker*3] == "ForeheadRightX"):
		FOREHEAD_RIGHT = marker
	elif(labels[marker*3] == "ForeheadLeftX"):
		FOREHEAD_LEFT = marker
	elif(labels[marker*3] == "ForeheadX"):
		FOREHEAD = marker
	elif(labels[marker*3] == "CheekRightX"):
		CHEEK_RIGHT = marker
	elif(labels[marker*3] == "CheekLeftX"):
		CHEEK_LEFT = marker
	elif(labels[marker*3] == "EyeRightX"):
		EYE_RIGHT = marker
	elif(labels[marker*3] == "EyeLeftX"):
		EYE_LEFT = marker
	elif(labels[marker*3] == "EyebrowRightX"):
		EYEBROW_RIGHT = marker
	elif(labels[marker*3] == "EyebrowLeftX"):
		EYEBROW_LEFT = marker
	elif(labels[marker*3] == "NoseRightX"):
		NOSE_RIGHT = marker
	elif(labels[marker*3] == "NoseLeftX"):
		NOSE_LEFT = marker
	elif(labels[marker*3] == "MouthRightX"):
		MOUTH_RIGHT = marker
	elif(labels[marker*3] == "MouthLeftX"):
		MOUTH_LEFT = marker
	elif(labels[marker*3] == "MouthUpX"):
		MOUTH_UP = marker
	elif(labels[marker*3] == "MouthDownX"):
		MOUTH_DOWN = marker
	elif(labels[marker*3] == "ChinX"):
		CHIN = marker


next(reader)
print(num_lines)
X_temp = np.zeros((num_lines,MARKER*3))
for i in range(2,num_lines):
    X_temp[i,:] = next(reader)[0:MARKER*3]

X = X_temp[2:,:]
print(X.shape)

#Put data in the right format (sequence of movement)
num = (X.shape[0]/FPS) / TIME #compute number of lines
size = MARKER*3*FREQ
X_std = np.zeros((num+1,MARKER*3*FREQ*TIME))
print(X_std.shape)

iter = 0
for i in range(0,num-1):
    for j in range(0,FREQ*TIME):
       	X_std[i][j*MARKER*3:(j+1)*MARKER*3] = X[iter]
        iter += FPS/FREQ
	#print(iter)


#Compute the PCA
sklearn_pca = sklearnPCA(n_components=COMP)
sklearn_pca =sklearn_pca.fit(X_std)

#Transform the data
Y = sklearn_pca.transform(X_std)


#example of generation of motion
Y_test = np.random.rand(2,COMP)
X_test =  sklearn_pca.inverse_transform(Y_test)
printDataChunk(X_test[0], labels, "generated")


#Print the explained variance for each composant
cum_var_exp = np.cumsum(sklearn_pca.explained_variance_ratio_)
with plt.style.context('seaborn-whitegrid'):
     plt.figure(figsize=(6, 10))
     plt.bar(range(COMP), sklearn_pca.explained_variance_ratio_, alpha=0.5, align='center',label='individual explained variance')
     plt.step(range(COMP), cum_var_exp, where='mid',color ='r',label='cumulative explained variance')
     plt.ylabel('Explained variance ratio')
     plt.xlabel('Principal components')
     plt.legend(loc='best')
     plt.tight_layout()
     plt.savefig("graph/bar")

import csv
from matplotlib import pyplot as plt
import numpy as np
import sys

#Draw e sequence of frame based on a processed csv file (given by autoEncoder.py)
#input Csv file
#output Images of frames
#use : "python drawSeq.py folder" OR "python drawSeq.py folder1 folder2" if 2 picture per frame

FREQ = 30 #number of iter for 1 second
TIME = 2 #number of seconds
MARKER = 16
NAME = "../Autoencoder/seq/"
NAME2 = "../Autoencoder/seq/"

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


def printDataChunk(data, labels,name):
    X = np.zeros((MARKER,TIME*FREQ))
    Y = np.zeros((MARKER,TIME*FREQ))
    color = ['bo', 'go','ro', 'co','mo', 'yo','ko', 'bv','gv','rv', 'cv','mv', 'yv','kv','gs','rs', 'cs']
    for i in range(0,MARKER):
        for j in range(TIME*FREQ):
            X[i][j] = data[j][i*3 + 1]
            Y[i][j] = data[j][i*3 + 2]

    with plt.style.context('seaborn-whitegrid'):
        
        for iter in range(0,TIME*FREQ):
            plt.figure(figsize=(10, 10))
            #plt.axis([-0.35, 0.85, -0.1, 1.1])
            for marker in range(0,MARKER):
				plt.plot(X[marker,iter],Y[marker,iter], color[marker], label = labels[marker*3 +1])

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

def printDoubleDataChunk(data1, data2, labels,name):
    X = np.zeros((MARKER,TIME*FREQ))
    Y = np.zeros((MARKER,TIME*FREQ))
    X2 = np.zeros((MARKER,TIME*FREQ))
    Y2 = np.zeros((MARKER,TIME*FREQ))
    color = ['bo', 'go','ro', 'co','mo', 'yo','ko', 'bv','gv','rv', 'cv','mv', 'yv','kv','gs','rs', 'cs']
    for i in range(0,MARKER):
        for j in range(TIME*FREQ):
            X[i][j] = data[j][i*3 + 1]
            Y[i][j] = data[j][i*3 + 2]
            X2[i][j] = data2[j][i*3 + 1]
            Y2[i][j] = data2[j][i*3 + 2]

    with plt.style.context('seaborn-whitegrid'):
        
        for iter in range(0,TIME*FREQ):
            fig = plt.figure(figsize=(20, 10))
            plt.axis([-0.35, 0.85, -0.1, 1.1])
            fig.add_subplot(1, 2, 1)
            for marker in range(0,MARKER):
				plt.plot(X[marker,iter],Y[marker,iter], color[marker], label = labels[marker*3 +1])

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

            fig.add_subplot(1, 2, 2)
            for marker in range(0,MARKER):
				plt.plot(X2[marker,iter],Y2[marker,iter], color[marker])

            #Draw the mouth
            plt.plot([X2[MOUTH_UP][iter], X2[MOUTH_LEFT][iter]], [Y2[MOUTH_UP][iter], Y2[MOUTH_LEFT][iter]],color='r', linewidth=2.0)
            plt.plot([X2[MOUTH_UP][iter], X2[MOUTH_RIGHT][iter]], [Y2[MOUTH_UP][iter], Y2[MOUTH_RIGHT][iter]],color='r', linewidth=2.0)
            plt.plot([X2[MOUTH_DOWN][iter], X2[MOUTH_LEFT][iter]], [Y2[MOUTH_DOWN][iter], Y2[MOUTH_LEFT][iter]],color='r', linewidth=2.0)
            plt.plot([X2[MOUTH_DOWN][iter], X2[MOUTH_RIGHT][iter]], [Y2[MOUTH_DOWN][iter], Y2[MOUTH_RIGHT][iter]],color='r', linewidth=2.0)
            
            #draw the eyes
            l1 = (Y2[EYEBROW_RIGHT][iter] - Y2[EYE_RIGHT][iter]) /2
            plt.plot([X2[EYE_RIGHT][iter], X2[EYE_RIGHT][iter]+l1], [Y2[EYE_RIGHT][iter], Y2[EYEBROW_RIGHT][iter]-3*l1/2],color='r', linewidth=2.0)
            plt.plot([X2[EYE_RIGHT][iter], X2[EYE_RIGHT][iter]-l1], [Y2[EYE_RIGHT][iter], Y2[EYEBROW_RIGHT][iter]-3*l1/2],color='r', linewidth=2.0)
            plt.plot([X2[EYEBROW_RIGHT][iter], X2[EYE_RIGHT][iter]+l1], [Y2[EYEBROW_RIGHT][iter]-l1, Y2[EYEBROW_RIGHT][iter]-3*l1/2],color='r', linewidth=2.0)
            plt.plot([X2[EYEBROW_RIGHT][iter], X2[EYE_RIGHT][iter]-l1], [Y2[EYEBROW_RIGHT][iter]-l1, Y2[EYEBROW_RIGHT][iter]-3*l1/2],color='r', linewidth=2.0)
            plt.plot([X2[EYE_LEFT][iter], X2[EYE_LEFT][iter]+l1], [Y2[EYE_LEFT][iter], Y2[EYEBROW_LEFT][iter]-3*l1/2],color='r', linewidth=2.0)
            plt.plot([X2[EYE_LEFT][iter], X2[EYE_LEFT][iter]-l1], [Y2[EYE_LEFT][iter], Y2[EYEBROW_LEFT][iter]-3*l1/2],color='r', linewidth=2.0)
            plt.plot([X2[EYEBROW_LEFT][iter], X2[EYE_LEFT][iter]+l1], [Y2[EYEBROW_LEFT][iter]-l1, Y2[EYEBROW_LEFT][iter]-3*l1/2],color='r', linewidth=2.0)
            plt.plot([X2[EYEBROW_LEFT][iter], X2[EYE_LEFT][iter]-l1], [Y2[EYEBROW_LEFT][iter]-l1, Y2[EYEBROW_LEFT][iter]-3*l1/2],color='r', linewidth=2.0)
            
            plt.ylabel('Y position')
            plt.xlabel('X position')
            plt.legend(loc='best')
            plt.tight_layout()

            plt.savefig("graph/graph"+name + str(iter))
            plt.close()


#read the data
f = open(NAME + str(sys.argv[1]) +".csv", 'rb')

#put data in matrix
reader = csv.reader(f)
labels = next(reader)
f.close()

for marker in range(0, MARKER):
	if(labels[marker*3 + 1] == "ForeheadRightX"):
		FOREHEAD_RIGHT = marker
	elif(labels[marker*3 + 1] == "ForeheadLeftX"):
		FOREHEAD_LEFT = marker
	elif(labels[marker*3 + 1] == "ForeheadX"):
		FOREHEAD = marker
	elif(labels[marker*3 + 1] == "CheekRightX"):
		CHEEK_RIGHT = marker
	elif(labels[marker*3 + 1] == "CheekLeftX"):
		CHEEK_LEFT = marker
	elif(labels[marker*3 + 1] == "EyeRightX"):
		EYE_RIGHT = marker
	elif(labels[marker*3 + 1] == "EyeLeftX"):
		EYE_LEFT = marker
	elif(labels[marker*3 + 1] == "EyebrowRightX"):
		EYEBROW_RIGHT = marker
	elif(labels[marker*3 + 1] == "EyebrowLeftX"):
		EYEBROW_LEFT = marker
	elif(labels[marker*3 + 1] == "NoseRightX"):
		NOSE_RIGHT = marker
	elif(labels[marker*3 + 1] == "NoseLeftX"):
		NOSE_LEFT = marker
	elif(labels[marker*3 + 1] == "MouthRightX"):
		MOUTH_RIGHT = marker
	elif(labels[marker*3 + 1] == "MouthLeftX"):
		MOUTH_LEFT = marker
	elif(labels[marker*3 + 1] == "MouthUpX"):
		MOUTH_UP = marker
	elif(labels[marker*3 + 1] == "MouthDownX"):
		MOUTH_DOWN = marker
	elif(labels[marker*3 + 1] == "ChinX"):
		CHIN = marker

if(len(sys.argv) == 2):
    data = np.loadtxt(NAME + str(sys.argv[1]) +".csv", delimiter=',', skiprows=1)[:,0:]
    printDataChunk(data, labels, "test")
elif(len(sys.argv) == 3):
    data = np.loadtxt(NAME + str(sys.argv[1]) +".csv", delimiter=',', skiprows=1)[:,0:]
    data2 = np.loadtxt(NAME + str(sys.argv[2]) +".csv", delimiter=',', skiprows=1)[:,0:]
    printDoubleDataChunk(data, data2, labels, "test")

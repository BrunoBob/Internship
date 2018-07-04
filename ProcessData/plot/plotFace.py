print("import")

import csv
import matplotlib.pyplot as plt
import numpy as np
import math

FILE_PATH = "../../Data/Bruno_1_juin/"
START_LINE = 581
END_LINE = 583
#52021 1781
FREQ = 1
NB_MARKER = 19
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

#Plot 2D graph function
def plot2D(X, Y, Z, num):
	color = ['bo', 'go','ro', 'co','mo', 'yo','ko', 'bv','gv','rv', 'cv','mv', 'yv','kv','gs','rs', 'cs','ms', 'ys','ks',]

	for graphNb in range(0,num):
		fig = plt.figure()
		plt.axis([-250, 250, -200, 120])
		#plt.axis([-200, 200, 900, 1300])
		
		for marker in range(0,NB_MARKER):
			plt.plot(X[graphNb][marker],Y[graphNb][marker], color[marker])

		name = FILE_PATH + "motionPlot2D/" + str(graphNb*FREQ) + ".png"
		print(name)
		fig.savefig(name)
		plt.close(fig)

#Plot 3D graph function
def plot3D(X, Y, Z, num):
	for graphNb in range(0,num):
		fig = plt.figure()
		plt.axis([-100, 100, 1000, 1300])
		for marker in range(0,NB_MARKER):
			plt.scatter(X[graphNb][marker],Y[graphNb][marker], math.sqrt(Z[graphNb][marker]*Z[graphNb][marker]))

		name = FILE_PATH + "motionPlot3D/" + str(graphNb*FREQ) + ".png"
		print(name)
		fig.savefig(name)
		plt.close(fig)

def correctMovement(X, Y, Z, num):
	
	#Get references vectors and normalisations
	Vinit = [X[0][REF_UP] - X[0][REF_CENTER], Y[0][REF_UP] - Y[0][REF_CENTER], Z[0][REF_UP] - Z[0][REF_CENTER]]
	Dinit = np.linalg.norm(Vinit)
	Vinit = Vinit / Dinit

	Vinit2 = [X[0][REF_LEFT] - X[0][REF_CENTER], Y[0][REF_LEFT] - Y[0][REF_CENTER], Z[0][REF_LEFT] - Z[0][REF_CENTER]]
	Dinit2 = np.linalg.norm(Vinit2)
	Vinit2 = Vinit2 / Dinit2
	
	#for each image
	for graphNb in range(1,num):

		#Get first current vector and normalisation
		Vcur = [X[graphNb][REF_UP] - X[graphNb][REF_CENTER], Y[graphNb][REF_UP] - Y[graphNb][REF_CENTER], Z[graphNb][REF_UP] - Z[graphNb][REF_CENTER]]
		Dcur = np.linalg.norm(Vcur)
		Vcur = Vcur / Dcur

		#Compute first the rotation matrix between vectro Vinit AND vCUR
		identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
		cross = np.cross(Vcur,Vinit)
		dot = np.dot(Vcur,Vinit)
		norm = np.linalg.norm(cross)
		ssc = np.array([(0, -cross[2], cross[1]), (cross[2], 0, -cross[0]), (-cross[1], cross[0], 0)])
		ssc2 = np.matmul(ssc, ssc)
		R = (1-dot)/(norm*norm)
		R = ssc2 * R
		R = ssc + R
		R = identity + R

		#Remove the first rotation for each marker
		for marker in range(0,NB_MARKER):
			point = [X[graphNb][marker], Y[graphNb][marker], Z[graphNb][marker]]
			#print(np.matrix(point))
			point = np.matmul(R,point)
			#print(np.matrix(point))
			X[graphNb][marker] = point[0]
			Y[graphNb][marker] = point[1]
			Z[graphNb][marker] = point[2]

		
		#Get second current vector and normalisation
		Vcur2 = [X[graphNb][REF_LEFT] - X[graphNb][REF_CENTER], Y[graphNb][REF_LEFT] - Y[graphNb][REF_CENTER], Z[graphNb][REF_LEFT] - Z[graphNb][REF_CENTER]]
		Dcur2 = np.linalg.norm(Vcur2)
		Vcur2 = Vcur2 / Dcur2

		#Compute second the rotation matrix between vectro Vinit AND vCUR
		identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
		cross = np.cross(Vcur2,Vinit2)
		dot = np.dot(Vcur2,Vinit2)
		norm = np.linalg.norm(cross)
		ssc = np.array([(0, -cross[2], cross[1]), (cross[2], 0, -cross[0]), (-cross[1], cross[0], 0)])
		ssc2 = np.matmul(ssc, ssc)
		R = (1-dot)/(norm*norm)
		R = ssc2 * R
		R = ssc + R
		R = identity + R

		#Remove the second rotation for each marker
		for marker in range(0,NB_MARKER):
			point = [X[graphNb][marker], Y[graphNb][marker], Z[graphNb][marker]]
			#print(np.matrix(point))
			point = np.matmul(R,point)
			#print(np.matrix(point))
			X[graphNb][marker] = point[0]
			Y[graphNb][marker] = point[1]
			Z[graphNb][marker] = point[2]

		#remove translation
		defX = X[graphNb][REF_CENTER] #- X[0][REF_CENTER_COL]
		defY = Y[graphNb][REF_CENTER] #- Y[0][REF_CENTER_COL]
		defZ = Z[graphNb][REF_CENTER] #- Z[0][REF_CENTER_COL]
		for marker in range(0,NB_MARKER):
			X[graphNb][marker] = X[graphNb][marker] - defX
			Y[graphNb][marker] = Y[graphNb][marker] - defY
			Z[graphNb][marker] = Z[graphNb][marker] - defZ


print("start")

#Read the csv file
f = open(FILE_PATH + 'motion.csv', 'rb')
reader = csv.reader(f)

#Get the data
num = (END_LINE - START_LINE) / FREQ
X = np.zeros((num,NB_MARKER))
Y = np.zeros((num,NB_MARKER))
Z = np.zeros((num,NB_MARKER))

#process the header
for iter in range(0,4):
	line = next(reader)

markerCheck = 0
for marker in range(0, NB_MARKER):
	if(line[marker*3 + 2] == "MarkerSet:RefUp"):
		REF_UP = marker
		markerCheck += 1
	elif(line[marker*3 + 2] == "MarkerSet:RefLeft"):
		REF_LEFT = marker
		markerCheck += 1
	elif(line[marker*3 + 2] == "MarkerSet:RefCenter"):
		REF_CENTER = marker
		markerCheck += 1
	elif(line[marker*3 + 2] == "MarkerSet:ForeheadRight"):
		FOREHEAD_RIGHT = marker
		markerCheck += 1
	elif(line[marker*3 + 2] == "MarkerSet:ForeheadLeft"):
		FOREHEAD_LEFT = marker
		markerCheck += 1
	elif(line[marker*3 + 2] == "MarkerSet:Forehead"):
		FOREHEAD = marker
		markerCheck += 1
	elif(line[marker*3 + 2] == "MarkerSet:CheekRight"):
		CHEEK_RIGHT = marker
		markerCheck += 1
	elif(line[marker*3 + 2] == "MarkerSet:CheekLeft"):
		CHEEK_LEFT = marker
		markerCheck += 1
	elif(line[marker*3 + 2] == "MarkerSet:EyeRight"):
		EYE_RIGHT = marker
		markerCheck += 1
	elif(line[marker*3 + 2] == "MarkerSet:EyeLeft"):
		EYE_LEFT = marker
		markerCheck += 1
	elif(line[marker*3 + 2] == "MarkerSet:EyebrowRight"):
		EYEBROW_RIGHT = marker
		markerCheck += 1
	elif(line[marker*3 + 2] == "MarkerSet:EyebrowLeft"):
		EYEBROW_LEFT = marker
		markerCheck += 1
	elif(line[marker*3 + 2] == "MarkerSet:NoseRight"):
		NOSE_RIGHT = marker
		markerCheck += 1
	elif(line[marker*3 + 2] == "MarkerSet:NoseLeft"):
		NOSE_LEFT = marker
		markerCheck += 1
	elif(line[marker*3 + 2] == "MarkerSet:MouthRight"):
		MOUTH_RIGHT = marker
		markerCheck += 1
	elif(line[marker*3 + 2] == "MarkerSet:MouthLeft"):
		MOUTH_LEFT = marker
		markerCheck += 1
	elif(line[marker*3 + 2] == "MarkerSet:MouthUp"):
		MOUTH_UP = marker
		markerCheck += 1
	elif(line[marker*3 + 2] == "MarkerSet:MouthDown"):
		MOUTH_DOWN = marker
		markerCheck += 1
	elif(line[marker*3 + 2] == "MarkerSet:Chin"):
		CHIN = marker
		markerCheck += 1

if(markerCheck == NB_MARKER):
	print("All marker found")
else:
	print("Some marker missing")

#Go to fisrt line of data
for iter in range(iter+1,START_LINE):
	line = next(reader)

#Read the data and save them in a matrix
for row in range(START_LINE, END_LINE , FREQ):
	line = next(reader)
	while(line[2]==""):
		line = next(reader)

	for axe in range(0,NB_MARKER):
		if(line[axe*3 +2] != ""):
			X[((row-START_LINE) / FREQ) ][axe] = float(line[axe*3 + 2])
		if(line[axe*3 +3] != ""):
			Y[(row-START_LINE) / FREQ][axe] = float(line[axe*3 + 3])
		if(line[axe*3 +4] != ""):
			Z[(row-START_LINE) / FREQ][axe] = float(line[axe * 3 + 4])
	if(reader.line_num + FREQ <= END_LINE):
		for iter in range(0,FREQ-1):
			line = next(reader)

print("Creat graph")
#plot the data
correctMovement(X,Y,Z,num)
plot2D(X,Y,Z,num)

	
print('end')
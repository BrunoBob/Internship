print("import")

import csv
import matplotlib.pyplot as plt
import numpy as np
import math

FILE_PATH = "../../Data/Alexandre/"
START_LINE = 2330
END_LINE = 36648
#581 52021 1781
FREQ = 2
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
		plt.axis([-0.35, 0.85, -0.1, 1.1])
		#plt.axis([-200, 200, 900, 1300])
		
		#Plot all marker
		for marker in range(0,NB_MARKER):
			plt.plot(X[graphNb][marker],Y[graphNb][marker], color[marker])

		#Draw the mouth
		plt.plot([X[graphNb][MOUTH_UP], X[graphNb][MOUTH_LEFT]], [Y[graphNb][MOUTH_UP], Y[graphNb][MOUTH_LEFT]],color='r', linewidth=2.0)
		plt.plot([X[graphNb][MOUTH_UP], X[graphNb][MOUTH_RIGHT]], [Y[graphNb][MOUTH_UP], Y[graphNb][MOUTH_RIGHT]],color='r', linewidth=2.0)
		plt.plot([X[graphNb][MOUTH_DOWN], X[graphNb][MOUTH_LEFT]], [Y[graphNb][MOUTH_DOWN], Y[graphNb][MOUTH_LEFT]],color='r', linewidth=2.0)
		plt.plot([X[graphNb][MOUTH_DOWN], X[graphNb][MOUTH_RIGHT]], [Y[graphNb][MOUTH_DOWN], Y[graphNb][MOUTH_RIGHT]],color='r', linewidth=2.0)

		#draw the eyes
		l1 = (Y[graphNb][EYEBROW_RIGHT] - Y[graphNb][EYE_RIGHT]) /2
		plt.plot([X[graphNb][EYE_RIGHT], X[graphNb][EYE_RIGHT]+l1], [Y[graphNb][EYE_RIGHT], Y[graphNb][EYEBROW_RIGHT]-3*l1/2],color='r', linewidth=2.0)
		plt.plot([X[graphNb][EYE_RIGHT], X[graphNb][EYE_RIGHT]-l1], [Y[graphNb][EYE_RIGHT], Y[graphNb][EYEBROW_RIGHT]-3*l1/2],color='r', linewidth=2.0)
		plt.plot([X[graphNb][EYEBROW_RIGHT], X[graphNb][EYE_RIGHT]+l1], [Y[graphNb][EYEBROW_RIGHT]-l1, Y[graphNb][EYEBROW_RIGHT]-3*l1/2],color='r', linewidth=2.0)
		plt.plot([X[graphNb][EYEBROW_RIGHT], X[graphNb][EYE_RIGHT]-l1], [Y[graphNb][EYEBROW_RIGHT]-l1, Y[graphNb][EYEBROW_RIGHT]-3*l1/2],color='r', linewidth=2.0)
		plt.plot([X[graphNb][EYE_LEFT], X[graphNb][EYE_LEFT]+l1], [Y[graphNb][EYE_LEFT], Y[graphNb][EYEBROW_LEFT]-3*l1/2],color='r', linewidth=2.0)
		plt.plot([X[graphNb][EYE_LEFT], X[graphNb][EYE_LEFT]-l1], [Y[graphNb][EYE_LEFT], Y[graphNb][EYEBROW_LEFT]-3*l1/2],color='r', linewidth=2.0)
		plt.plot([X[graphNb][EYEBROW_LEFT], X[graphNb][EYE_LEFT]+l1], [Y[graphNb][EYEBROW_LEFT]-l1, Y[graphNb][EYEBROW_LEFT]-3*l1/2],color='r', linewidth=2.0)
		plt.plot([X[graphNb][EYEBROW_LEFT], X[graphNb][EYE_LEFT]-l1], [Y[graphNb][EYEBROW_LEFT]-l1, Y[graphNb][EYEBROW_LEFT]-3*l1/2],color='r', linewidth=2.0)

		#Save the data in a picture
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

	maxX = maxY = maxZ = -10000000
	minX = minY = minZ = 10000000
	
	#remove noise
	for graphNb in range(2,num):
		for marker in range(0,NB_MARKER):
			if((X[graphNb][marker] - X[graphNb-1][marker]) < -100 or (X[graphNb][marker] - X[graphNb-1][marker]) > 100):
				X[graphNb][marker] = X[graphNb-1][marker]
			if((Y[graphNb][marker] - Y[graphNb-1][marker]) < -100 or (Y[graphNb][marker] - Y[graphNb-1][marker]) > 100):
				Y[graphNb][marker] = Y[graphNb-1][marker]
			if((Z[graphNb][marker] - Z[graphNb-1][marker]) < -100 or (Z[graphNb][marker] - Z[graphNb-1][marker]) > 100):
				Z[graphNb][marker] = Z[graphNb-1][marker]
	

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

		#search for max and min
		for marker in range(0,NB_MARKER):
			if(X[graphNb][marker] > maxX):
				maxX = X[graphNb][marker]
			if(X[graphNb][marker] < minX):
				minX = X[graphNb][marker]
			if(Y[graphNb][marker] > maxY):
				maxY = Y[graphNb][marker]
			if(Y[graphNb][marker] < minY):
				minY = Y[graphNb][marker]
			if(Z[graphNb][marker] > maxZ):
				maxZ = Z[graphNb][marker]
			if(Z[graphNb][marker] < minZ):
				minZ = Z[graphNb][marker]
	
	#scale the data between 0 and 1
	for graphNb in range(1,num):
		for marker in range(0,NB_MARKER):
			X[graphNb][marker] = ((X[graphNb][marker] - minX)/ (maxX-minX))*0.5
			Y[graphNb][marker] = (Y[graphNb][marker] - minY)/ (maxY-minY)
			Z[graphNb][marker] = (Z[graphNb][marker] - minZ)/ (maxZ-minZ)

def saveData(names, X, Y, Z, num):
	print("Saving the data")
	f = open(FILE_PATH + "processedMotion.csv", 'wb')
	writer = csv.writer(f)
	writer.writerow(names)
	data = np.zeros((NB_MARKER-3)*3)
	for line in range(0,num):
		for marker in range(0, NB_MARKER-3):
			data[marker*3]=X[line][marker+3]
			data[marker*3+1]=Y[line][marker+3]
			data[marker*3+2]=Z[line][marker+3]
		writer.writerow(data)

print("start")

#Read the csv file
f = open(FILE_PATH + 'motion.csv', 'rb')
reader = csv.reader(f)

#Get the data
num = (END_LINE - START_LINE) / FREQ
X = np.zeros((num+1,NB_MARKER))
Y = np.zeros((num+1,NB_MARKER))
Z = np.zeros((num+1,NB_MARKER))

#process the header
for iter in range(0,4):
	line = next(reader)

names = np.chararray((NB_MARKER-3)*3,itemsize = 20)
for name in range(0, NB_MARKER-3):
	names[name*3] = line[name*3 +11][10:] + 'X'
	names[name*3+1] = line[name*3 +11][10:] + 'Y'
	names[name*3+2] = line[name*3 +11][10:] + 'Z'
print(names)

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
			X[(row-START_LINE) / FREQ ][axe] = float(line[axe*3 + 2])
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
#plot2D(X,Y,Z,num)
saveData(names,X,Y,Z,num)
	
print('end')
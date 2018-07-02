print("import")

import csv
import matplotlib.pyplot as plt
import numpy as np
import math

FILE_PATH = "../../Data/Bruno_1_juin/"
START_LINE = 581
END_LINE = 1781
#52021 1781
FREQ = 10
NB_MARKER = 19
REF_CENTER_COL = 2
REF_LEFT_COL = 1
REF_UP_COL = 0

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
	
	#Get reference vector and normalisation
	Vinit = [X[0][REF_UP_COL] - X[0][REF_CENTER_COL], Y[0][REF_UP_COL] - Y[0][REF_CENTER_COL], Z[0][REF_UP_COL] - Z[0][REF_CENTER_COL]]
	Dinit = np.linalg.norm(Vinit)
	Vinit = Vinit / Dinit
	
	#for each image
	for graphNb in range(1,num):

		#Get current vector and normalisation
		Vcur = [X[graphNb][REF_UP_COL] - X[graphNb][REF_CENTER_COL], Y[graphNb][REF_UP_COL] - Y[graphNb][REF_CENTER_COL], Z[graphNb][REF_UP_COL] - Z[graphNb][REF_CENTER_COL]]
		Dcur = np.linalg.norm(Vcur)
		Vcur = Vcur / Dcur

		#Compute the rotation matrix between vectro Vinit AND vCUR
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

		#Remove the rotation for each marker
		for marker in range(0,NB_MARKER):
			point = [X[graphNb][marker], Y[graphNb][marker], Z[graphNb][marker]]
			#print(np.matrix(point))
			point = np.matmul(R,point)
			#print(np.matrix(point))
			X[graphNb][marker] = point[0]
			Y[graphNb][marker] = point[1]
			Z[graphNb][marker] = point[2]

		#remove translation
		defX = X[graphNb][REF_CENTER_COL] #- X[0][REF_CENTER_COL]
		defY = Y[graphNb][REF_CENTER_COL] #- Y[0][REF_CENTER_COL]
		defZ = Z[graphNb][REF_CENTER_COL] #- Z[0][REF_CENTER_COL]
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

#Ignore the header lines
for iter in range(0,START_LINE-1):
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

#Center the data on reference point

""" for row in range(0,num):
	defX = X[row][REF_CENTER_COL] #- X[0][REF_CENTER_COL]
	defY = Y[row][REF_CENTER_COL] #- Y[0][REF_CENTER_COL]
	for axe in range(0,NB_MARKER):
		X[row][axe] = X[row][axe] - defX
		Y[row][axe] = Y[row][axe] - defY """

print("Creat graph")
#plot the data
correctMovement(X,Y,Z,num)
plot2D(X,Y,Z,num)

	
print('end')
print("import")

import csv
import matplotlib.pyplot as plt
import numpy as np
import math

#Plot 2D graph function
def plot2D():
	color = ['bo', 'go','ro', 'co','mo', 'yo','ko', 'bv','gv','rv', 'cv','mv', 'yv','kv','gs','rs', 'cs','ms', 'ys','ks',]
	for graphNb in range(0,num):
		fig = plt.figure()
		plt.axis([-250, 250, -200, 120])
		for marker in range(0,NB_MARKER):
			plt.plot(X[graphNb][marker],Y[graphNb][marker], color[marker])

		name = FILE_PATH + "motionPlot2D/" + str(graphNb*FREQ) + ".png"
		print(name)
		fig.savefig(name)
		plt.close(fig)

#Plot 3D graph function
def plot3D():
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
	max = 0
	min = 10000
	v1 = 0
	for row in range(0,num):
		dist12 = math.sqrt(((X[row][REF_CENTER_COL]-X[row][REF_UP_COL])**2) +((Y[row][REF_CENTER_COL]-Y[row][REF_UP_COL])**2) +((Z[row][REF_CENTER_COL]-Z[row][REF_UP_COL])**2))
		if(dist12 > max):
			max = dist12
		if(dist12 < min):
			min = dist12
	print(min)
	print(max)
	print(max-min)

FILE_PATH = "../../Data/Bruno_1_juin/"
START_LINE = 581
END_LINE = 52021
#52021 1781
FREQ = 100#6
NB_MARKER = 19
REF_CENTER_COL = 2
REF_LEFT_COL = 1
REF_UP_COL = 0

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
print(FREQ)
#Read the data and save them in a matrix
for row in range(START_LINE, END_LINE , FREQ):
	line = next(reader)
	while(line[2]==""):
		line = next(reader)
	#print(row)
	for axe in range(0,NB_MARKER):
		if(line[axe*3 +2] != ""):
			X[((row-START_LINE) / FREQ) - 1][axe] = float(line[axe*3 + 2])
		if(line[axe*3 +3] != ""):
			Y[(row-START_LINE) / FREQ - 1][axe] = float(line[axe*3 + 3])
		if(line[axe*3 +4] != ""):
			Z[(row-START_LINE) / FREQ - 1][axe] = float(line[axe * 3 + 4])
	
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
#plot2D()

	
print('end')
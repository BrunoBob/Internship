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
		plt.axis([-150, 50, 1100, 1200])
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
		#plt.axis([-150, 50, 1100, 1200])
		for marker in range(0,NB_MARKER):
			plt.scatter(X[graphNb][marker],Y[graphNb][marker], math.sqrt(Z[graphNb][marker]*Z[graphNb][marker]))

		name = FILE_PATH + "motionPlot3D/" + str(graphNb*FREQ) + ".png"
		print(name)
		fig.savefig(name)
		plt.close(fig)


FILE_PATH = "../../Data/Test1/"
START_LINE = 8
END_LINE = 248
FREQ = 6
NB_MARKER = 4

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
for row in range(START_LINE, END_LINE-1 , FREQ):
	line = next(reader)
	while(line[2]==""):
		line = next(reader)

	for axe in range(0,NB_MARKER):
		if(line[axe*3 +2] != ""):
			X[row / FREQ - 1][axe] = float(line[axe*3 + 2])
		if(line[axe*3 +3] != ""):
			Y[row / FREQ - 1][axe] = float(line[axe*3 + 3])
		if(line[axe*3 +4] != ""):
			Z[row / FREQ - 1][axe] = float(line[axe * 3 + 4])
	
	if(reader.line_num + FREQ <= END_LINE):	
		for iter in range(0,FREQ-1):
			line = next(reader)

print("Creat graph")
#plot the data
plot3D()

	
print('end')
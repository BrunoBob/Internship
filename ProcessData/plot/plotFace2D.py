print("import")

import csv
import matplotlib.pyplot as plt
import numpy as np

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

for graphNb in range(0,num):
	fig = plt.figure()
	plt.axis([-150, 50, 1100, 1200])
	plt.plot(X[graphNb][0],Y[graphNb][0], 'bo')
	plt.plot(X[graphNb][1],Y[graphNb][1], 'go')
	plt.plot(X[graphNb][2],Y[graphNb][2], 'ro')
	plt.plot(X[graphNb][3],Y[graphNb][3], 'co')
	
	name = FILE_PATH + "motionPlot2D/" + str(graphNb*FREQ) + ".png"
	print(name)
	fig.savefig(name)
	plt.close(fig)
	
print('end')
		
			

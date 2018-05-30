import csv
import matplotlib.pyplot as plt

FILE_PATH = "../../Data/Test1/"
START_LINE = 8
END_LINE = 1000
FREQ = 100

#Read the csv file
f = open(FILE_PATH + 'motion.csv', 'rb')
reader = csv.reader(f)

#Get the data
for iter in range(0,START_LINE-1):
	line = next(reader)

for row in range(0, END_LINE-1 , FREQ):
	print(row)
	line = next(reader)
	while(line[2]==""):
		line = next(reader)
	fig = plt.figure()
	plt.axis([-150, 50, 1100, 1200])
	plt.plot(float(line[2]),float(line[3]), 'bo')
	plt.plot(float(line[5]),float(line[6]), 'go')
	plt.plot(float(line[8]),float(line[9]), 'ro')
	plt.plot(float(line[11]),float(line[12]), 'co')
	
	name = FILE_PATH + "motionPlot2D/" +str(reader.line_num) + ".png"
	print(name)
	fig.savefig(name)
	plt.close(fig)
	for iter in range(0,FREQ-1):
		line = next(reader)
	
		
			

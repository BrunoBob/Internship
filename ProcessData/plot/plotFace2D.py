import csv
import matplotlib
import matplotlib.pyplot as plt
f = open('../../Data/Test1/motion.csv', 'rb')
reader = csv.reader(f)
for row in range(0, 1000):
	line = next(reader)
	if(reader.line_num >= 8 and line[2]!=""):
		fig = plt.figure()
		plt.axis([-150, 50, 1100, 1200])
		plt.plot(float(line[2]),float(line[3]), 'bo')
		plt.plot(float(line[5]),float(line[6]), 'go')
		plt.plot(float(line[8]),float(line[9]), 'ro')
		plt.plot(float(line[11]),float(line[12]), 'co')
		#print(reader.line_num)
		#print(line[2])
		name = "../../Data/Test1/motionPlot2D/"+str(reader.line_num)+".png"
		fig.savefig(name)
		plt.close(fig)
		for test in range(0,9):
			line = next(reader)
		
			

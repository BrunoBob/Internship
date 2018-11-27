import csv
from matplotlib import pyplot as plt
import numpy as np
import math

FREQ = 30 #number of iter for 1 second
FPS = 60 #fps of input data
TIME = 2 #number of seconds
MARKER = 16
PATH = "../../Data/Jiro/"

#read the data
f = open(PATH + 'processedMotion.csv', 'rb')
num_lines = sum(1 for line in open(PATH + 'processedMotion.csv'))

#put data in matrix
reader = csv.reader(f)
labels = next(reader)

next(reader)
print(num_lines)
X_temp = np.zeros((num_lines,MARKER*3))
for i in range(2,num_lines):
    X_temp[i,:] = next(reader)[0:MARKER*3]

X = X_temp[2:,:]
print(X.shape)

num = (X.shape[0]/FPS) / TIME #compute number of lines
size = MARKER*3*FREQ
X_std = np.zeros((num+1,MARKER*3*FREQ*TIME))
print(X_std.shape)

iter = 0
for i in range(0,num-1):
    for j in range(0,FREQ*TIME):
       	X_std[i][j*MARKER*3:(j+1)*MARKER*3] = X[iter]
        iter += FPS/FREQ

f_write = open(PATH + "dataToLearn.csv", 'wb')
writer = csv.writer(f_write)
writer.writerow(labels)
#current_data = np.zeros(MARKER*3 + 1)
for line in range(0,X_std.shape[0]-2):
    writer.writerow(X_std[line])
f.close()
f_write.close()
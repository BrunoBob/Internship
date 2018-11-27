from matplotlib import pyplot as plt
import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
import csv
import random

#Run an autoencoder to try to learn how to generate motion file

MARKER = 16
TIME = 2
FREQ = 30
NAME = '../../Data/Jiro/dataToLearn.csv'

def saveData(labels, data, name):
    f = open("seq/" + name, 'w')
    writer = csv.writer(f)
    label = ["" for x in range(MARKER*3+1)]
    label[0] = "time_seconds"
    label[1:MARKER*3] = labels
    writer.writerow(label)
    current_data = np.zeros(MARKER*3 + 1)
    for line in range(0,TIME * FREQ):
        current_data[0] = float(line)/float(FREQ)
        current_data[1:MARKER*3+1] = data[line * MARKER * 3 : (line+1) * MARKER * 3]
        writer.writerow(current_data)
    f.close()

all_images = np.loadtxt(NAME, delimiter=',', skiprows=1)[:,0:]
print(all_images.shape[0])

f = open(NAME, 'rt')
reader = csv.reader(f)
labels = next(reader)
f.close()

# printing something that actually looks like an image
#plt.imshow(mnist.train.images[0].reshape(28,28),  cmap='Greys')
#plt.show()

# Deciding how many nodes wach layer should have
n_nodes_inpl = 2880  #input
n_nodes_hl1  = 720  #encoder
n_nodes_hl2  = 100  #encoder
n_nodes_hl3  = 3  #encoder
n_nodes_hl4  = 100  #decoder
n_nodes_hl5  = 720  #decoder
n_nodes_outl = 2880  #decoder

batch_size = 1 # how many images to use together for training
hm_epochs = 100  # how many times to go through the entire dataset
tot_images = all_images.shape[0] # total number of images

# image with shape 784 goes in
input_layer = tf.placeholder('float', [None, n_nodes_inpl])

# first hidden layer has 784*196 weights and 196 biases
hidden_1_layer_vals = {
'weights':tf.Variable(tf.random_normal([n_nodes_inpl,n_nodes_hl1])),
'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))  }

# Second hidden layer has 196*49 weights and 49 biases
hidden_2_layer_vals = {
'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))  }

# Third hidden layer has 49*196 weights and 196 biases
hidden_3_layer_vals = {
'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))  }

hidden_4_layer_vals = {
'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_nodes_hl4])),
'biases':tf.Variable(tf.random_normal([n_nodes_hl4]))  }

hidden_5_layer_vals = {
'weights':tf.Variable(tf.random_normal([n_nodes_hl4,n_nodes_hl5])),
'biases':tf.Variable(tf.random_normal([n_nodes_hl5]))  }

# Output hidden layer has 196*784 weights and 784 biases
output_layer_vals = {
'weights':tf.Variable(tf.random_normal([n_nodes_hl5,n_nodes_outl])),
'biases':tf.Variable(tf.random_normal([n_nodes_outl])) }

# multiply output of input_layer wth a weight matrix and add biases
layer_1 = tf.nn.sigmoid(
       tf.add(tf.matmul(input_layer,hidden_1_layer_vals['weights']),
       hidden_1_layer_vals['biases']))

# multiply output of layer_1 wth a weight matrix and add biases
layer_2 = tf.nn.sigmoid(
       tf.add(tf.matmul(layer_1,hidden_2_layer_vals['weights']),
       hidden_2_layer_vals['biases']))

# multiply output of layer_2 wth a weight matrihttp://ibriquel.free.fr/enseignement/DCA/x and add biases
layer_3 = tf.nn.sigmoid(
       tf.add(tf.matmul(layer_2,hidden_3_layer_vals['weights']),
       hidden_3_layer_vals['biases']))

layer_4 = tf.nn.sigmoid(
       tf.add(tf.matmul(layer_3,hidden_4_layer_vals['weights']),
       hidden_4_layer_vals['biases']))

layer_5 = tf.nn.sigmoid(
       tf.add(tf.matmul(layer_4,hidden_5_layer_vals['weights']),
       hidden_5_layer_vals['biases']))

decod1 = tf.placeholder('float', [None, n_nodes_hl3])

decod2 = tf.nn.sigmoid(
       tf.add(tf.matmul(decod1,hidden_4_layer_vals['weights']),
       hidden_4_layer_vals['biases']))

decod3 = tf.nn.sigmoid(
       tf.add(tf.matmul(decod2,hidden_5_layer_vals['weights']),
       hidden_5_layer_vals['biases']))

gen_layer = tf.matmul(decod3,output_layer_vals['weights']) +  output_layer_vals['biases']

# multiply output of layer_5 wth a weight matrix and add biases
output_layer = tf.matmul(layer_5,output_layer_vals['weights']) +  output_layer_vals['biases']

# output_true shall have the original image for error calculations
output_true = tf.placeholder('float', [None, n_nodes_inpl])

# define our cost function
meansq =    tf.reduce_mean(tf.square(output_layer - output_true))

# define our optimizer
learn_rate = 0.1   # how fast the model should learn
optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(meansq)

# initialising stuff and starting the session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# defining batch size, number of epochs and learning rate

# running the model for a 1000 epochs taking 100 images in batches
# total improvement is printed out after each epoch
for epoch in range(hm_epochs):
	epoch_loss = 0    # initializing error as 0
	for i in range(int(tot_images/batch_size)):
		epoch_x = all_images[ i*batch_size : (i+1)*batch_size ]
		_, c = sess.run([optimizer, meansq],feed_dict={input_layer: epoch_x, output_true: epoch_x})
		epoch_loss += c

	print('Epoch', epoch, '/', hm_epochs, 'loss:',epoch_loss)


randNum = random.randint(0, tot_images)

saveData(labels, all_images[randNum], "init1.csv")


output_any_image1 = sess.run(output_layer, feed_dict={input_layer:[all_images[randNum]]})


code = sess.run(layer_3, feed_dict={input_layer:[all_images[1]]})
decode = sess.run(gen_layer, feed_dict={decod1:[code[0]]})


saveData(labels, output_any_image1[0], "new1.csv")

genTest = np.random.rand(1,n_nodes_hl3)
gen1 = sess.run(gen_layer, feed_dict={decod1:[genTest[0]]})
saveData(labels, gen1[0], "gen1.csv")

for iter in range(tot_images-1):
    num = sess.run(layer_3, feed_dict={input_layer:[all_images[iter]]})
    
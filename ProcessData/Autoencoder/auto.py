from matplotlib import pyplot as plt
import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# printing something that actually looks like an image
#plt.imshow(mnist.train.images[0].reshape(28,28),  cmap='Greys')
#plt.show()

# Deciding how many nodes wach layer should have
n_nodes_inpl = 784  #input
n_nodes_hl1  = 196  #encoder
n_nodes_hl2  = 49  #encoder
n_nodes_hl3  = 196  #decoder
n_nodes_outl = 784  #decoder

# image with shape 784 goes in
input_layer = tf.placeholder('float', [None, 784])

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

# Output hidden layer has 196*784 weights and 784 biases
output_layer_vals = {
'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_nodes_outl])),
'biases':tf.Variable(tf.random_normal([n_nodes_outl])) }

# multiply output of input_layer wth a weight matrix and add biases
layer_1 = tf.nn.relu(
       tf.add(tf.matmul(input_layer,hidden_1_layer_vals['weights']),
       hidden_1_layer_vals['biases']))

# multiply output of layer_1 wth a weight matrix and add biases
layer_2 = tf.nn.relu(
       tf.add(tf.matmul(layer_1,hidden_2_layer_vals['weights']),
       hidden_2_layer_vals['biases']))

# multiply output of layer_2 wth a weight matrix and add biases
layer_3 = tf.nn.relu(
       tf.add(tf.matmul(layer_2,hidden_3_layer_vals['weights']),
       hidden_3_layer_vals['biases']))

# multiply output of layer_3 wth a weight matrix and add biases
output_layer = tf.nn.relu(
		tf.add(tf.matmul(layer_3,output_layer_vals['weights']),
        output_layer_vals['biases']))

# output_true shall have the original image for error calculations
output_true = tf.placeholder('float', [None, 784])

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
batch_size = 100  # how many images to use together for training
hm_epochs =10    # how many times to go through the entire dataset
tot_images = 60000 # total number of images
# running the model for a 1000 epochs taking 100 images in batches
# total improvement is printed out after each epoch
for epoch in range(hm_epochs):
	epoch_loss = 0    # initializing error as 0
	for i in range(int(mnist.train.num_examples/batch_size)):
		epoch_x, epoch_y = mnist.train.next_batch(batch_size)
		_, c = sess.run([optimizer, meansq],feed_dict={input_layer: epoch_x, output_true: epoch_x})
		epoch_loss += c

	print('Epoch', epoch, '/', hm_epochs, 'loss:',epoch_loss)


# pick any image
any_image = mnist.train.images[999]
# run it though the autoencoder
output_any_image = sess.run(output_layer, feed_dict={input_layer:[any_image]})
# run it though just the encoder
encoded_any_image = sess.run(layer_1, feed_dict={input_layer:[any_image]})

# print the encoding
print(encoded_any_image)

# print the original image
fig=plt.figure(figsize=(8, 8))
fig.add_subplot(1, 2, 1)
plt.imshow(any_image.reshape(28,28),  cmap='Greys')
fig.add_subplot(1, 2, 2)
plt.imshow(any_image.reshape(28,28),  cmap='Greys')
plt.show()

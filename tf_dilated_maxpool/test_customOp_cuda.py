import tensorflow as tf
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



dilated_maxpooling_gpu_module = tf.load_op_library('/home/jzuern/tensorflow/tensorflow/core/user_ops/dilated_maxpooling_gpu.so')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
gpu_options.allow_growth=True
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# image = random.random((2,10,10,3)) # Format:  NHWC -- [batch, height, width, channels]
image1 = mpimg.imread('bird1.png')
image2 = mpimg.imread('bird2.png')
image_shape = image1.shape

images = np.zeros(shape=(2,image_shape[0],image_shape[1],3),dtype=float)
images[0,:,:,:] = image1
images[1,:,:,:] = image2
nBatches = images.shape[0]
nChannels = images.shape[3]

print "images shape: ", images.shape
window_size = [1,5,5,1]
strides = [1,2,2,1]

# Creates a graph
with tf.device('/gpu:0'):
  bilateral  = dilated_maxpooling_gpu_module.dilated_max_pooling(images, ksize=window_size, strides=strides,padding='VALID',dilation_rate=1)


output = sess.run(bilateral) # Runs the op.
print "input: ", images
print "output: ", output

## iterate through batches and show every batch entry:
b = 0 # batch entry number

fig = plt.figure()
fig = plt.figure()
subplot=fig.add_subplot(1,2,1)
imgplot = plt.imshow(images[b,:,:,:],interpolation='None')
subplot.set_title('Original')
a=fig.add_subplot(1,2,2)
imgplot = plt.imshow(output[b,:,:,:],interpolation='None')
subplot.set_title('after MaxPool')

plt.show()

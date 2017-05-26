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
height = image_shape[0]
width = image_shape[1]


images = np.zeros(shape=(2,height,width,3),dtype=float)
images[0,:,:,:] = image1
images[1,:,:,:] = image2


print "images shape: ", images.shape
window_size = [1,2,2,1]
strides = [1,2,2,1]

# Creates a graph
with tf.device('/gpu:0'):
  bilateral  = dilated_maxpooling_gpu_module.dilated_max_pooling(images, ksize=window_size, strides=strides,padding='VALID',dilation_rate=2)


output = sess.run(bilateral) # Runs the op.
print "input: ", images
print "output: ", output



######### BATCH 0
fig = plt.figure()
a=fig.add_subplot(1,2,1)
imgplot = plt.imshow(images[0,:,:,:],interpolation='None') ## v2

a.set_title('Original')
a=fig.add_subplot(1,2,2)
imgplot = plt.imshow(output[0,:,:,:],interpolation='None')## v2
a.set_title('after MaxPool')

######### BATCH 1
fig = plt.figure()
a=fig.add_subplot(1,2,1)
imgplot = plt.imshow(images[0,:,:,:],interpolation='None') ## v2

a.set_title('Original')
a=fig.add_subplot(1,2,2)
imgplot = plt.imshow(output[0,:,:,:],interpolation='None') ## v2
a.set_title('after MaxPool')

plt.show()

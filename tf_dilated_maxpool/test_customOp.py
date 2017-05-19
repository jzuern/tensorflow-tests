import tensorflow as tf
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

dilated_maxpooling_module = tf.load_op_library('/home/jzuern/tensorflow/tensorflow/core/user_ops/dilated_maxpooling.so')

with tf.Session(''):
  image = mpimg.imread('bird.png')
  image = np.expand_dims(image, axis=0)
  #image = random.random((1,8,8,3)) # Format:  NHWC -- [batch, height, width, channels]

  window_size = [1,2,2,1]
  strides = [1,2,2,1]
  result = dilated_maxpooling_module.dilated_max_pool(image, ksize=window_size, strides=strides,padding='VALID',dilation_rate=2)
  output = result.eval()

  fig = plt.figure()
  a=fig.add_subplot(1,2,1)
  imgplot = plt.imshow(image[0,:,:,0],cmap='gist_gray',interpolation='None')
  # imgplot = plt.imshow(image[0,:,:,:],interpolation='None')

  a.set_title('Original')
  a=fig.add_subplot(1,2,2)
  imgplot = plt.imshow(output[0,:,:,0],cmap='gist_gray',interpolation='None')
  # imgplot = plt.imshow(output[0,:,:,:],interpolation='None')
  a.set_title('after MaxPool')

  plt.show()

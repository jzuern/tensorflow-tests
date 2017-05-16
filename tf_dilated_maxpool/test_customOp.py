import tensorflow as tf
from numpy import random
import matplotlib.pyplot as plt

dilated_maxpooling_module = tf.load_op_library('/home/jzuern/tensorflow/tensorflow/core/user_ops/dilated_maxpooling.so')

with tf.Session(''):

  image = random.random((1,10,10,1)) # Format:  NHWC -- [batch, height, width, channels]
  print "input image: ", image

  k = 2
  result = dilated_maxpooling_module.dilated_max_pool(image, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='VALID',dilation_rate=1)
  output = result.eval()

  print "output image: ", output

  fig = plt.figure()
  a=fig.add_subplot(1,2,1)
  imgplot = plt.imshow(image[0,:,:,0],cmap='binary',interpolation='None')
  a.set_title('Original')
  a=fig.add_subplot(1,2,2)
  imgplot = plt.imshow(output[0,:,:,0],cmap='binary',interpolation='None')
  a.set_title('after MaxPool')

  plt.show()

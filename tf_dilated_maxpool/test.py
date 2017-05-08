import tensorflow as tf
from pylab import imshow, show, get_cmap
from numpy import random

dilated_maxpooling_module = tf.load_op_library('/home/jzuern/tensorflow/tensorflow/core/user_ops/dilated_maxpooling.so')
zero_out_module = tf.load_op_library('/home/jzuern/tensorflow/tensorflow/core/user_ops/zero_out.so')

with tf.Session(''):
  # print dilated_maxpooling_module.dilated_maxpooling([[1, 2], [3, 4]]).eval()
  image = random.random((1,10,10,3)) # Format:  NHWC -- [batch, height, width, channels]
  print "input image: ", image
  k = 2
  result = dilated_maxpooling_module.dilated_max_pool(image, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')
  print "output image: ", result.eval()

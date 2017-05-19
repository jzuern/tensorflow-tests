import tensorflow as tf
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

dilated_maxpooling_gpu_module = tf.load_op_library('/home/jzuern/tensorflow/tensorflow/core/user_ops/dilated_maxpooling_gpu.so')




gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
gpu_options.allow_growth=True
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

image = random.random((1,4,4,1)) # Format:  NHWC -- [batch, height, width, channels]
window_size = [1,2,2,1]
strides = [1,2,2,1]

# Creates a graph
with tf.device('/gpu:0'):
  bilateral  = dilated_maxpooling_gpu_module.dilated_max_pooling(image, ksize=window_size, strides=strides,padding='VALID',dilation_rate=1)


out = sess.run(bilateral) # Runs the op.
print "input: ", image
print "output: ", out

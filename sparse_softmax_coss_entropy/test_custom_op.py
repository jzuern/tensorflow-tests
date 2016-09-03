
import tensorflow as tf

zero_out_module = tf.load_op_library('/home/jzuern/tf_installation/tensorflow/tensorflow/core/user_ops/zero_out.so')


with tf.Session(''):
  zero_out_module.zero_out([[1, 2], [3, 4]]).eval()

# Prints
array([[1, 0],
       [0, 0]], dtype=int32)

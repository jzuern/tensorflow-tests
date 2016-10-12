
import tensorflow as tf

zero_out_module = tf.load_op_library('/home/jzuern/tf_installation/tensorflow/tensorflow/core/user_ops/zero_out.so')


with tf.Session(''):
  a = zero_out_module.zero_out([[5, 2], [3, 4]]).eval()

print a

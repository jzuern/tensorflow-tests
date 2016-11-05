import tensorflow as tf

# load shared object containing custom OpKernel
gpu_test_module = tf.load_op_library('/home/jzuern/tf_installation/tensorflow/bazel-bin/tensorflow/core/user_ops/cuda_op_kernel.so')

zero_out_module = tf.load_op_library('/home/jzuern/tf_installation/tensorflow/bazel-bin/tensorflow/core/user_ops/zero_out.so')
with tf.Session(''):
  zero_out_module.zero_out([[1, 2], [3, 4]]).eval()





#################################################################
######### GPU ###################################################
#################################################################

print('testing TensorFlow on a GPU device')


# Creates a graph.
with tf.device('/gpu:0'):
  ones = tf.ones([10, 10], tf.int32)
  added_one  = gpu_test_module.add_one(ones)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print sess.run(added_one)


#################################################################
######### CPU ###################################################
#################################################################

# print('testing TensorFlow on a CPU device')
#
# # Creates a graph.
# with tf.device('/cpu:0'):
#   ones = tf.ones([10, 10], tf.int32)
#   to_zero  = cpu_test_mod.zero_out([[1, 2], [3, 4]])
#
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # Runs the op.
# print sess.run(to_zero)

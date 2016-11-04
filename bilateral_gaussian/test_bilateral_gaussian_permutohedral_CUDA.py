import tensorflow as tf

# load shared object containing custom OpKernel
cuda_mod = tf.load_op_library('/home/jzuern/tf_installation/tensorflow/tensorflow/core/user_ops/cuda_op_kernel.so')


# 
# random_tensor = tf.random_normal([50,50,3], mean=0.5, stddev=0.3, dtype=tf.float32)
#
# print(random_tensor)
#
# with tf.Session(''):
#   added_one  = cuda_mod.add_one(random_tensor).eval()
#
#
# print(added_one)

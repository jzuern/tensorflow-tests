import tensorflow as tf
import numpy as np

# load shared object containing custom OpKernel
permutohedral_module = tf.load_op_library('/home/jzuern/tf_installation/tensorflow/tensorflow/core/user_ops/bilateral_gaussian_permutohedral.so')


# spatial standard deviation:
stdv_spat = 5.

# color standard deviation
stdv_col = 0.125

# input tensor
shape = [20,20,3] # 20px x 20px image with 3 color channels (RGB)

input = tf.fill(shape, 128)


print("input image:")
print(input)


# calculate loss bilateral gaussian blur
blur = permutohedral_module.permutohedral_bilateral_gaussian_blur(input,stdv_spat,stdv_col)

# Launch the default graph.
sess = tf.Session()

# calculate result
[blurred] = sess.run(blur)

# print result
print("blurred:")
print(blurred)

# Close the Session when we're done.
sess.close()

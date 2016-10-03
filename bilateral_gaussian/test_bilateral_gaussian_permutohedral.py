import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.image as mpimg

# load shared object containing custom OpKernel
permutohedral_module = tf.load_op_library('/home/jzuern/tf_installation/tensorflow/tensorflow/core/user_ops/bilateral_gaussian_permutohedral.so')



# read input image
image = mpimg.imread('input.png')

# show input image
plt.imshow(image)
plt.show()


# spatial standard deviation:
stdv_spat = 5.0

# color standard deviation
stdv_col = 0.125

# convert mpimg image to tensorflow tensor object
image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)


# calculate bilateral gaussian blur
blur = permutohedral_module.bilateral_gaussian_permutohedral(image_tensor, stdv_spat, stdv_col)

# Launch the default graph.
sess = tf.Session()

# calculate result
blurred = sess.run(blur)

# show output image
plt.imshow(blurred)
plt.show()

# Close the Session when we're done.
sess.close()

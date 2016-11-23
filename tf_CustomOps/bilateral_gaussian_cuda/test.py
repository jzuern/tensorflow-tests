import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

print('-----Testing TensorFlow on a GPU device-----')


# load shared object containing custom OpKernel
bilateral_module = tf.load_op_library('/home/jzuern/tf_installation/tensorflow/bazel-bin/tensorflow/core/user_ops/bilateral_gaussian_cuda/bilateral_gaussian_permutohedral_cuda.so')

# read input image
image = mpimg.imread('input_square_small.png')

# spatial standard deviation:
stdv_spat = 5.0

# color standard deviation
stdv_col = 1.0

# convert mpimg image to tensorflow tensor object
image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
random_tensor = tf.random_normal([400,100,3], mean=0.5, stddev=0.1, dtype=tf.float32)

print("Input Tensor: \n")
print(random_tensor.eval(session=tf.Session('')))
print("\n\n")
plt.imshow(image_tensor.eval(session=tf.Session('')))  #show dog pic
plt.show()



# Creates a graph.
with tf.device('/gpu:0'):
  bilateral  = bilateral_module.bilateral_gaussian_permutohedral_cuda(image_tensor,stdv_spat,stdv_col)

# make session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Runs the op.
out = sess.run(bilateral)

# show output image
print("output image:")
print(out)
plt.imshow(out)
plt.show()

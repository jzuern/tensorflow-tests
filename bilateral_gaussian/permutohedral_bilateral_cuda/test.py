import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

print('-----Testing TensorFlow on a GPU device-----')

# load shared object containing custom OpKernel

bilateral_module = tf.load_op_library('/home/jzuern/tensorflow/bazel-bin/tensorflow/core/user_ops/bilateral_gaussian_cuda/bilateral_gaussian_permutohedral_cuda.so')

# read input image
image = mpimg.imread('input_square_small.png')

# spatial standard deviation:
stdv_spat = 5.0

# color standard deviation
stdv_col = 1.0

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
gpu_options.allow_growth=True
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# convert mpimg image to tensorflow tensor object
# im = tf.convert_to_tensor(image, dtype=tf.float32)
im = tf.random_normal([512,512,64], mean=0.5, stddev=0.2, dtype=tf.float32)
ref = tf.random_normal([512,512,3], mean=0.5, stddev=0.2, dtype=tf.float32)

reverse = False # False if filter, True if gradient of filter

# plt.imshow(image_tensor.eval(session=sess))  #show pic
# plt.show()

# Creates a graph
with tf.device('/gpu:0'):
  bilateral  = bilateral_module.bilateral_gaussian_permutohedral_cuda(im,ref,stdv_spat,stdv_col, reverse)

for i in range(0,3):
  print "i =",i
  out = sess.run(bilateral) # Runs the op.



# show output image
plt.imshow(out)
plt.show()

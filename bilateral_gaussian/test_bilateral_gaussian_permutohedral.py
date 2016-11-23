import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

# load shared object containing custom OpKernel
blur_module = tf.load_op_library('/home/jzuern/tf_installation/tensorflow/tensorflow/core/user_ops/bilateral_gaussian_permutohedral.so')


# read input image
image = mpimg.imread('input_orig.png')

# spatial standard deviation:
stdv_spat = 5.0

# color standard deviation
stdv_col = 1.0

# convert mpimg image to tensorflow tensor object
image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
random_tensor = tf.random_normal([50,50,3], mean=0.5, stddev=0.3, dtype=tf.float32)


print (image_tensor)
print (random_tensor)

# show input image
with tf.Session(''):
    plt.imshow(random_tensor.eval()) #show white noise pic
    plt.imshow(image_tensor.eval())  #show dog pic
    plt.show()



with tf.Session(''):
  blurred  = blur_module.bilateral_gaussian_permutohedral(image_tensor, stdv_spat, stdv_col).eval()
  blurred_grad = blur_module.bilateral_gaussian_permutohedral_grad(image_tensor, stdv_spat, stdv_col).eval()



# show output image
plt.imshow(blurred)
plt.show()

plt.imshow(blurred_grad)
plt.show()

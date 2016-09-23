import tensorflow as tf
import numpy as np

# load shared object containing custom OpKernel
sparse_weighted_module = tf.load_op_library('/home/jzuern/tf_installation/tensorflow/tensorflow/core/user_ops/sparse_weighted.so')


# log-probabilities

# outer vector size: batch size (4)
# inner vector size: number of classes (2)
logits = tf.constant([[1.0, 2.0],
                      [3.0, 4.0],
                      [3.0, 4.0],
                      [1.2, 4.3]])

 # correct labels for each batch entry (4 items)
labels = tf.constant([0,1,0,1])

# weight for each class (2 items)
weights = tf.constant([ 2.0 , 1.0])

# calculate loss with weighted classes
sw = sparse_weighted_module.sparse_weighted(logits,labels,weights)

# Launch the default graph.
sess = tf.Session()

# calculate result
[loss, grad] = sess.run(sw)

# print out result
print("loss:")
print(loss)

print("gradient:")
print(grad)

# sanity checks

# Close the Session when we're done.
sess.close()

import tensorflow as tf

sparse_weighted_module = tf.load_op_library('/home/jzuern/tf_installation/tensorflow/tensorflow/core/user_ops/sparse_weighted.so')



# log-probabilities
logits = tf.constant([[1.0, 2.0, 5.2],
                      [3.0, 4.0, 1.1],
                      [1.2, 4.3, 3.2]])

 # correct labels for each row in logits
labels = tf.constant([1,1,1])


#weight for positive targets
pos_weight = 4.0



sw = sparse_weighted_module.sparse_weighted(logits,labels,pos_weight)


# Launch the default graph.
sess = tf.Session()

# calculate result
result = sess.run(sw)

# print out result
print(result)


# Close the Session when we're done.
sess.close()

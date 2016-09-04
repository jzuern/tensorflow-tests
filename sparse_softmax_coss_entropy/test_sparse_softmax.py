import tensorflow as tf

logits = tf.constant([[1.0, 2.0], [3.0, 4.0]])
labels = tf.constant([0,1])

sm = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,labels)

# Launch the default graph.
sess = tf.Session()

result = sess.run(sm)
print(result)

# ==> [[ 12.]]

# Close the Session when we're done.
sess.close()

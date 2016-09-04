import tensorflow as tf

logits = tf.constant([[1.0, 2.0], [3.0, 4.0]])
targets = tf.constant([[1.0, 2.0], [3.0, 4.0]])
pos_weight = 4.0

wc = tf.nn.weighted_cross_entropy_with_logits(logits,targets,pos_weight)


# Launch the default graph.
sess = tf.Session()

result = sess.run(wc)
print(result)


# Close the Session when we're done.
sess.close()

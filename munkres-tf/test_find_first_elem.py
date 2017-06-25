# test with tensorflow r1.0
import tensorflow as tf

val = 3
m = tf.placeholder(tf.int32)
""" in each row must be >1 entry of "val"  """
m_feed = [[val, 0,    0,    0,   0 ,   0,   0],
          [0  , val, val,   0,    0   ,0   ,0],
          [0  ,   0, 0,    val,   0   ,0   ,0],
          [0  ,   0, 0,    0,    val ,  0,  0],
          [0  ,   0, val,   0,    0 ,  val, 0],
          [0  ,   0, 0,     0,   val , 0,  0],
          [0  ,   0, 0,     val,  0 ,  0,  0]]

tmp_indices = tf.where(tf.equal(m, val))
col_indx = tf.segment_min(tmp_indices[:, 1], tmp_indices[:, 0])
row_indx = tf.cast(tf.range(7),tf.int64)

marked = tf.Variable(tf.zeros([7,7],dtype=tf.int32))
ind = tf.stack([row_indx,col_indx], axis=1)
values = tf.ones([7],dtype=tf.int32)
shape = [7, 7]  # The shape of the corresponding dense tensor, same as `marked`.
delta = tf.SparseTensor(ind, values, shape)
result = marked + tf.sparse_tensor_to_dense(delta)



init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    print sess.run(ind,feed_dict={m: m_feed})
    print sess.run(result,feed_dict={m: m_feed})
    print(sess.run(row_indx, feed_dict={m: m_feed})) # [2, 0, 1]
    print(sess.run(col_indx, feed_dict={m: m_feed})) # [2, 0, 1]

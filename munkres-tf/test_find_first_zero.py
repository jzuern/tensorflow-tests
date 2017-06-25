# test with tensorflow r1.0
import tensorflow as tf

val = 1
m = tf.placeholder(tf.int32)
row_covered = tf.placeholder(tf.int32)
col_covered = tf.placeholder(tf.int32)

""" in each row must be >1 entry of "val"  """
m_feed = [[0  , 0,  0,  0,    0 ,   0,  0],
          [0  , 0, 0,   0,    0    ,0  ,0],
          [0  , 0, 0,   val,   0    ,0  ,0],
          [0  , 0, 0,   0,   val ,  0,  0],
          [0  , 0, val, 0,    0 ,  val, 0],
          [0  , 0, 0,   0,   val , 0,   0],
          [0  , 0, 0,  val,   0 ,  0,   0]]


row_covered_feed = [0,0,1,1,0,0,1]
col_covered_feed = [0,0,1,0,1,0,1]

C_usable = tf.add(m,10)

row_matrix = tf.expand_dims(row_covered,0)
col_matrix = tf.expand_dims(col_covered,1)

row_m = tf.tile(row_matrix,tf.pack([7, 1]))
col_m = tf.tile(col_matrix,tf.pack([1, 7]))

C_new = tf.multiply(tf.multiply(m,row_m),col_m) # C' = C_ij * row_m_ij * col_m_ij

tmp_indices = tf.where(tf.equal(C_new, val))

row = tmp_indices[0,0]
col = tmp_indices[0,1]

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    print sess.run(C_usable,feed_dict={m: m_feed,row_covered: row_covered_feed, col_covered: col_covered_feed})
    print sess.run(C_new,feed_dict={m: m_feed,row_covered: row_covered_feed, col_covered: col_covered_feed})
    print sess.run(tmp_indices,feed_dict={m: m_feed,row_covered: row_covered_feed, col_covered: col_covered_feed})
    print sess.run(row,feed_dict={m: m_feed,row_covered: row_covered_feed, col_covered: col_covered_feed})
    print sess.run(col,feed_dict={m: m_feed,row_covered: row_covered_feed, col_covered: col_covered_feed})

import tensorflow as tf
import numpy as np


def init_matrix(shape):
    return tf.Variable(tf.zeros(shape))


def compute(matrix):



    # tf.while_loop(condition,body,
    # step = tf.case({tf.equal(step,1): __step1,
    #                      tf.equal(step,2): __step2,
    #                      tf.equal(step,3): __step3,
    #                      tf.equal(step,4): __step4,
    #                      tf.equal(step,5): __step5,
    #                      tf.equal(step,6): __step6},
    #                      default=__step1,exclusive=True)

    '''
    while (!done){
        switch(step){
            case 1:
                step = step_1(ref step);
                break;
            case 2:
                step = step_2(ref step);
                break;
            ...
            case 7:
                step = step_7(ref step);
                done = true;
                break;
        }
    }
    '''

    C = tf.identity(matrix) # make identical copy of matrix
    shape_C = C.get_shape()
    n = shape_C[0] # size of matrix

    col_covered = tf.Variable(tf.zeros([shape_C[0]]))
    row_covered = tf.Variable(tf.zeros([shape_C[0]]))
    marked = tf.zeros(shape_C)



    step,C = __step1(n,C)
    step,printout1 = __step2(marked,col_covered,row_covered,C)

    test =__find_smallest(C,col_covered,row_covered)

    return printout1,test

def __step1(n,C):

    C_min = tf.reduce_min(C,reduction_indices=[1])
    diff = C - C_min

    return tf.constant(2),diff # go to step 2

def __step2(marked,col_covered,row_covered,C):
    """find a zero in resulting matrix"""
    shape_C = C.get_shape()
    n = shape_C[0] # size of matrix

    # find indices for zeros in C
    tmp_indices_row = tf.where(tf.equal(C, 0))

    # get row indices
    col_indx = tf.segment_min(tmp_indices_row[:, 1], tmp_indices_row[:, 0])
    row_indx = tf.cast(tf.range(n),tf.int64)

    # set values in "marked" matrix
    marked = tf.Variable(tf.zeros(shape_C,dtype=tf.int32))
    ind = tf.stack([row_indx,col_indx], axis=1)
    values = tf.ones([n],dtype=tf.int32)
    delta = tf.SparseTensor(ind, values, shape_C)
    result = marked + tf.sparse_tensor_to_dense(delta)

    # no need to set row_covered or col_covered (they were not set)
    # row_covered,col_covered = __clear_covers(C.get_shape())

    return marked,C


def __step3(marked,col_covered,n):
    """
    Cover each column containing a starred zero. If K columns are
    covered, the starred zeros describe a complete set of unique
    assignments. In this case, Go to DONE, otherwise, Go to Step 4.
    """

    marked_reduced = tf.reduce_any(marked,0) # reduce along matrix columns TODO: correct axis?
    both = tf.concat(0, [marked_reduced, col_covered]) # join both tensors
    col_covered_new = tf.reduce_any(both,0) # TODO: correct axis?
    count = tf.reduce_sum(col_covered_new,axis=0) # TODO: correct axis?
    step = tf.cond(tf.greater_equal(count,n), 7, 4) # if true, goto step 7 (done), else, goto step 4

    return col_covered_new,step


def __find_prime_in_row(marked,row):
    """
    Find the first prime element in the specified row. Returns
    the column index, or -1 if no starred element was found.
    """

    marked_col = tf.squeeze(tf.gather(marked, col))
    idx_find = tf.where(tf.equal(marked_col, 2))

    try:
        col = tf.segment_min(idx_find)
        return col
    except Exception as e :
        return -1 # return col = -1 when we find now row containing a "1"


def __find_star_in_col(C,col):

    C_col = tf.squeeze(tf.gather(C, col))
    idx_find = tf.where(tf.equal(C_col, 1))

    try:
        row = tf.segment_min(idx_find)
        return row
    except Exception as e :
        return -1 # return row = -1 when we find now col containing a "1"

def __find_star_in_row(C,row):

    C_row = tf.squeeze(tf.gather(C, row))
    idx_find = tf.where(tf.equal(C_row, 1))

    try:
        col = tf.segment_min(idx_find)
        return col
    except Exception as e :
        return -1 # return col = -1 when we find now row containing a "1"


def __clear_covers(shape_C):
    """Clear all covered matrix cells"""

    row_covered = tf.Variable(tf.zeros([shape_C[0]]))
    col_covered = tf.Variable(tf.zeros([shape_C[1]]))

    return row_covered,col_covered

def __erase_primes(marked,C):

    indices =  tf.equal(C,fill(C.get_shape(), 2)) # compare
    marked = tf.scatter_update(marked, indices, 0)

    return marked

def __find_smallest(C,row_covered, col_covered):

    # first: filter out values of C that do not lie
    #         in row_covered or col_covered
    # must broadcast row_covered in dim 0
    # and broadcast col_covered in dim 1
    # and then filter
    C_shape = C.get_shape()
    n = C_shape[0]

    # add empty dimension to vector (--> matrix)
    row_matrix = tf.expand_dims(row_covered,1)
    col_matrix = tf.expand_dims(col_covered,1)


    row_m = tf.tile(row_matrix,tf.pack([n, 1]))
    col_m = tf.tile(col_matrix,tf.pack([n, 1]))

    #
    # first: use tf.argmin to find minimum of
    #         remaining tensor
    C_row = tf.tensordot(C,row_m,[[1], [0]])
    C_row_col = tf.tensordot(C_row,col_m,[[1], [0]])

    minval = tf.reduce_min(C_row_col,axis=None) # reduce along all two dimensions

    return minval,row_m



def __find_a_zero(C,row_covered,col_covered):
    """Find the first uncovered element with value 0"""

    C_shape = C.get_shape()
    n = C_shape[0]

    # add arbitrary value to each tensor element of C. this makes sorting out easier later
    C_usable = tf.add(C,10)

    row_matrix = tf.expand_dims(row_covered,0)
    col_matrix = tf.expand_dims(col_covered,1)

    row_m = tf.tile(row_matrix,tf.pack([n, 1]))
    col_m = tf.tile(col_matrix,tf.pack([1, n]))

    # eliminate all entries of C that do not
    C_filtered = tf.multiply(tf.multiply(C_usable,row_m),col_m) # C' = C_ij * row_m_ij * col_m_ij

    try:
        # find indices where C'_ij == 10
        indices = tf.where(tf.equal(C_filtered, 10))
        row = indices[0,0] # take first entry of indices
        col = indices[0,1]
    except Exception as e: # when nothing was found
        row = -1
        col = -1

    return (row,col)
################################################
################################################


matrix = tf.constant([[1, 2, 3], [4, 5,6], [7,8, 9]])


init = tf.global_variables_initializer()

result = compute(matrix)



# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    print(sess.run(result))


# Close the Session when we're done.
sess.close()

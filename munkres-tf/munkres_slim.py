#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

import sys
import numpy as np
import copy

def compute(cost_matrix):
    """
    Compute the indexes for the lowest-cost pairings between rows and
    columns in the database. Returns a list of (row, column) tuples
    that can be used to traverse the matrix.

    :Parameters:
        cost_matrix : list of lists
            The cost matrix. If this cost matrix is not square, it
            will be padded with zeros, via a call to ``pad_matrix()``.
            (This method does *not* modify the caller's matrix. It
            operates on a copy of the matrix.)

            **WARNING**: This code handles square and rectangular
            matrices. It does *not* handle irregular matrices.

    :rtype: list
    :return: A list of ``(row, column)`` tuples that describe the lowest
             cost path through the matrix

    """
    global C; C = copy.deepcopy(cost_matrix)
    global n; n = len(C)
    global original_length; original_length = len(cost_matrix)
    global original_width; original_width = len(cost_matrix[0])
    global row_covered; row_covered = [False for i in range(n)]
    global col_covered;col_covered = [False for i in range(n)]
    global Z0_r;Z0_r = 0
    global Z0_c;Z0_c = 0
    global path;path = __make_matrix(n * 2, 0)
    global marked;marked = __make_matrix(n, 0)

    done = False
    step = 1

    steps = { 1 : __step1,
              2 : __step2,
              3 : __step3,
              4 : __step4,
              5 : __step5,
              6 : __step6 }

    while not done:
        try:
            func = steps[step]
            step = func()
        except KeyError:
            done = True

    # Look for the starred columns
    results = []
    for i in range(original_length):
        for j in range(original_width):
            if marked[i][j] == 1:
                results += [(i, j)]

    return results


def __make_matrix(n, val):
    """Create an *n*x*n* matrix, populating it with the specific value."""
    _matrix = []
    for i in range(n):
        _matrix += [[val for j in range(n)]]
    return _matrix

def __step1():
    """
    For each row of the matrix, find the smallest element and
    subtract it from every element in its row. Go to Step 2.
    """
    global C

    for i in range(n):
        minval = min(C[i])
        # Find the minimum value for this row and subtract that minimum
        # from every element in the row.
        for j in range(n):
            C[i][j] -= minval
    return 2

def __step2():
    """
    Find a zero (Z) in the resulting matrix. If there is no starred
    zero in its row or column, star Z. Repeat for each element in the
    matrix. Go to Step 3.
    """
    global marked,col_covered,row_covered

    for i in range(n):
        for j in range(n):
            if (C[i][j] == 0) and \
                    (not col_covered[j]) and \
                    (not row_covered[i]):
                marked[i][j] = 1
                col_covered[j] = True
                row_covered[i] = True
                break

    __clear_covers()
    return 3

def __step3():
    """
    Cover each column containing a starred zero. If K columns are
    covered, the starred zeros describe a complete set of unique
    assignments. In this case, Go to DONE, otherwise, Go to Step 4.
    """
    global col_covered

    count = 0
    for i in range(n):
        for j in range(n):
            if marked[i][j] == 1 and not col_covered[j]:
                col_covered[j] = True
                count += 1

    if count >= n:
        step = 7 # done, all co
    else:
        step = 4

    print "col_covered: ",col_covered
    return step

def __step4():
    """
    Find a noncovered zero and prime it. If there is no starred zero
    in the row containing this primed zero, Go to Step 5. Otherwise,
    cover this row and uncover the column containing the starred
    zero. Continue in this manner until there are no uncovered zeros
    left. Save the smallest uncovered value and Go to Step 6.
    """
    print "starting step 4"
    global marked,row_covered,col_covered,Z0_r,Z0_c

    step = 0
    done = False
    row = -1
    col = -1
    star_col = -1
    while not done:
        (row, col) = __find_a_zero()

        if row < 0:
            done = True
            step = 6
        else:
            marked[row][col] = 2
            star_col = __find_star_in_row(row)
            if star_col >= 0:
                col = star_col
                row_covered[row] = True
                col_covered[col] = False
            else:
                done = True
                Z0_r = row
                Z0_c = col
                step = 5

    return step

def __step5():
    """
    Construct a series of alternating primed and starred zeros as
    follows. Let Z0 represent the uncovered primed zero found in Step 4.
    Let Z1 denote the starred zero in the column of Z0 (if any).
    Let Z2 denote the primed zero in the row of Z1 (there will always
    be one). Continue until the series terminates at a primed zero
    that has no starred zero in its column. Unstar each starred zero
    of the series, star each primed zero of the series, erase all
    primes and uncover every line in the matrix. Return to Step 3
    """
    count = 0
    global path

    path[count][0] = Z0_r
    path[count][1] = Z0_c
    done = False
    while not done:
        row = __find_star_in_col(path[count][1])
        if row >= 0:
            count += 1
            path[count][0] = row
            path[count][1] = path[count-1][1]
        else:
            done = True

        if not done:
            col = __find_prime_in_row(path[count][0])
            count += 1
            print "counter  = ", count
            path[count][0] = path[count-1][0]
            path[count][1] = col

    __convert_path(path, count)
    __clear_covers()
    __erase_primes()
    return 3

def __step6():
    """
    Add the value found in Step 4 to every element of each covered
    row, and subtract it from every element of each uncovered column.
    Return to Step 4 without altering any stars, primes, or covered
    lines.
    """
    global C

    minval = __find_smallest()
    events = 0 # track actual changes to matrix
    for i in range(n):
        for j in range(n):
            if row_covered[i]:
                C[i][j] += minval
                events += 1
            if not col_covered[j]:
                C[i][j] -= minval
                events += 1
            if row_covered[i] and not col_covered[j]:
                events -= 2 # change reversed, no real difference
    return 4

def __find_smallest():
    """Find the smallest uncovered value in the matrix."""
    minval = sys.maxsize
    for i in range(n):
        for j in range(n):
            if (not row_covered[i]) and (not col_covered[j]):
                if minval > C[i][j]:
                    minval = C[i][j]
    return minval

def __find_a_zero():
    """Find the first uncovered element with value 0"""
    row = -1
    col = -1
    i = 0
    done = False
    global C,row_covered,col_covered

    while not done:
        j = 0
        while True:
            if (C[i][j] == 0) and \
                    (not row_covered[i]) and \
                    (not col_covered[j]):
                row = i
                col = j
                done = True
            j += 1 # increment j
            if j >= n:
                break
        i += 1 # increment i
        if i >= n:
            done = True

    return (row, col)

def __find_star_in_row(row):
    """
    Find the first starred element in the specified row. Returns
    the column index, or -1 if no starred element was found.
    """
    global marked

    col = -1
    for j in range(n):
        if marked[row][j] == 1:
            col = j
            break

    return col

def __find_star_in_col(col):
    """
    Find the first starred element in the specified row. Returns
    the row index, or -1 if no starred element was found.
    """
    global marked

    row = -1
    for i in range(n):
        if marked[i][col] == 1:
            row = i
            break

    return row

def __find_prime_in_row(row):
    """
    Find the first prime element in the specified row. Returns
    the column index, or -1 if no starred element was found.
    """
    col = -1
    for j in range(n):
        if marked[row][j] == 2:
            col = j
            break

    return col

def __convert_path(path, count):
    global marked
    for i in range(count+1):
        if marked[path[i][0]][path[i][1]] == 1:
            marked[path[i][0]][path[i][1]] = 0
        else:
            marked[path[i][0]][path[i][1]] = 1

def __clear_covers():
    """Clear all covered matrix cells"""
    global row_covered,col_covered
    for i in range(n):
        row_covered[i] = False
        col_covered[i] = False

def __erase_primes():
    """Erase all prime markings"""
    global marked
    for i in range(n):
        for j in range(n):
            if marked[i][j] == 2:
                marked[i][j] = 0



# matrix = np.array(np.random.randint(20, size=(5, 5)))
matrix = [[5, 9, 1],
         [2, 3, 2],
         [8, 7, 4]]
print matrix
original_length = len(matrix)
original_width = len(matrix[0])

indexes = compute(matrix)

total = 0
for row, column in indexes:
    value = matrix[row][column]
    total += value
    print '(%d, %d) -> %d' % (row, column, value)
print 'total cost: %d' % total

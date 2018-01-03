#
# matrices.py
#
# Implement a representation of a typical matrix and associated
# operations. Matrices provide O(1) lookup given indices, which
# works great when we are storing many relations.
#


# The shape of a matrix
def shape(A):
    n_rows = len(A)
    n_cols = len(A[0]) if A else 0
    return n_rows, n_cols

# Get row vector
def get_row(A, i):
    return A[i]

# Get column vector
def get_column(A, j):
    return [row[j] for row in A]

# Generator using function
def make_matrix(n_rows, n_cols, entry_fn):
    return [[entry_fn(i, j)
        for j in range(n_cols)]
        for i in range(n_rows)]

# Identity matrix with diagonal of 1s
def make_identity_matrix(i, j):
    return make_matrix(i, j, lambda x,y: 1 if x==y else 0)


def main():

    # A typical matrix representation is a list of lists
    A = [[1,2,3],
         [4,5,6]]


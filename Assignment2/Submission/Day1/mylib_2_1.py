#row operations
def row_swap(A, i, j):
    A[i], A[j] = A[j], A[i]
    return A

def row_multiply(A, i, k):
    n = len(A[i])
    A[i] = [A[i][j]*k for j in range(n)]
    return A

def row_add(A, i, j, k):
    n = len(A[i])
    A[i] = [A[i][c] + A[j][c]*k for c in range(n)]    
    return A

#Creating Augmented matrix
def Augment(A, b):
    n = len(A)
    Aug_A = [A[i]+[b[i]] for i in range(n)]
    return Aug_A

#finidng max valu of column 1
def col1_max(A):
    n = len(A)
    col1 = [A[i][0] for i in range(n)]
    m = max(col1)
    j = col1.index(m)
    return j

#gauss jordan
def gauss_jordan(A, b):
    n = len(A)
    index = col1_max(A)
    Aug = Augment(A, b)
    row_swap(Aug, index, 0)
    for c in range(n):
        a = Aug[c][c]
        row_multiply(Aug, c, 1/a)
        for i in range(n):
            if i != c:    
                a = Aug[i][c]
                if a != 0:
                    row_add(Aug, i, c, -a)
    b = [Aug[i][n] for i in range(n)]                
    return {'Augmented Matrix': Aug, 
            'Solution': b}  
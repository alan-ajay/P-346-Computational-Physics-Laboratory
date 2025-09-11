import numpy as np

def vector(g, x):
    return [i(x) for i in g]

def norm(x):
    return np.sqrt(sum([i**2 for i in x]))

def multivar_fixed_point(g, guess, e = 1e-6):
    x = guess
    count = 0
    while True:  
        count += 1
        x0, x = np.array(x), np.array(vector(g, x))
        if norm(x - x0)/norm(x) < e:
                return x, count
        
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

def Id(n):
    X = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        X[i][i] = 1
    return X

#Creating Inverse Augmented matrix
def Augment_inv(A):
    n = len(A)          # number of rows
    X = Id(n)            # identity matrix
    Aug_A = [A[i] + X[i] for i in range(n)]
    return Aug_A

#finidng max valu of column 1
def col1_max(A):
    n = len(A)
    col1 = [A[i][0] for i in range(n)]
    m = max(col1)
    j = col1.index(m)
    return j

#gauss jordan
def Invert(A):
    n = len(A)
    index = col1_max(A)
    Aug = Augment_inv(A)
    row_swap(Aug, index, 0)
    for c in range(n):
        a = Aug[c][c]
        row_multiply(Aug, c, 1/a)
        for i in range(n):
            if i != c:    
                a = Aug[i][c]
                if a != 0:
                    row_add(Aug, i, c, -a)
    Ain = [row[n:] for row in Aug]              
    return Ain

def matrix_vector(A, b):
    N = len(A)
    Ab = []
    for i in range(N):
        s = 0
        for j in range(N):
            s += A[i][j] * b[j]
        Ab.append(s)
    return Ab



def Jacobian(J, x):
    n = len(J)
    return [[f(x) for f in J[i]] for i in range(n)]

def multivar_newton_raphson(f, J, guess, e = 1e-6):
    x = guess
    count = 0
    while True:  
        count += 1      
        f_val = vector(f, x) 
        Jin = Invert(Jacobian(J, x))
        y = np.array(matrix_vector(Jin, f_val))
        x = np.array(x)
        x0, x = x, x - y
        if norm(x - x0)/norm(x) < e:
                return x, count

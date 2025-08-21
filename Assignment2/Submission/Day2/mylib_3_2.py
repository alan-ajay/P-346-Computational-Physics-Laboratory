

def matrix_multiply(A, B):
    N = len(A)
    AB = []
    for i in range(N):
        L = []
        for j in range(N):
            sum = 0
            
            for k in range(N):
                sum += A[i][k]*B[k][j]
            L.append(round(sum, 1))    
        AB.append(L) 
    return AB


def LU_decompose(A):
    n = len(A)
    L, U = [[0 for _ in range(n)] for _ in range(n)], [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        L[i][i] = 1
    for j in range(n):
        U[0][j] = A[0][j]
    for j in range(n):
        for i in range(1, j+1):
            sum = 0
            for k in range(i):
                sum += L[i][k]*U[k][j]
            U[i][j] = A[i][j] - sum
        for i in range(j, n):
            sum = 0
            for k in range(j):
                sum += L[i][k]*U[k][j]
            L[i][j] = (A[i][j] - sum)/U[j][j]   
    return {'L' : L,
            'U' : U}   


def LU_solve(A, b):
    n = len(A)
    sol = LU_decompose(A)
    L = sol['L']
    U = sol ['U']
    A = []
    y = []

    for i in range(n):
        sum = 0
        for k in range(i):
            sum += L[i][k]*y[k]
        yi = b[i] - sum
        y.append(yi)
    x = [0 for _ in range(n)]
    for i in range(n-1, -1, -1):
        sum = 0
        for k in range(i+1, n):
            sum += U[i][k]*x[k]
        x[i] = (y[i] - sum)/U[i][i]
    return x

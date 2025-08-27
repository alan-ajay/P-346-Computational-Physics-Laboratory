import numpy as np

def Cholesky_decomposition(A):
    n = len(A)
    L = [[0 for i in range(n)] for i in range(n)]
    for i in range(n):
        sum1 = 0
        for j in range(i):
            sum1 += L[j][i]**2
        L[i][i] = np.sqrt(A[i][i] - sum1)
        for j in range(i+1, n):
            sum2 =0
            for k in range(i):
                sum2+=L[k][i]*L[k][j]
            L[i][j] = (A[i][j] - sum2)/(L[i][i])        
    return L  

def Cholesky_solve(A, b):
    n = len(A)
    U = Cholesky_decomposition(A)
    L = Transpose(U)
    A = []
    y = []

    for i in range(n):
        sum = 0
        for k in range(i):
            sum += L[i][k]*y[k]
        yi = (b[i] - sum)/L[i][i]
        y.append(yi)
    x = [0 for _ in range(n)]
    for i in range(n-1, -1, -1):
        sum = 0
        for k in range(i+1, n):
            sum += U[i][k]*x[k]
        x[i] = (y[i] - sum)/U[i][i]
    return x

def Transpose(A):
    n = len(A)
    L = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            L[i][j] = A[j][i]
    return L

def Jacobi_iter(A, b, e = 1e-6, x = None):
    n = len(A)
    Stop  = False
    x = [0 for _ in range(n)]
    count = 0
    while Stop  == False:
        count += 1
        xi = x.copy()
        for i in range(n):
            sum = 0
            for j in range(n):
                if j != i:
                    sum+= A[i][j]*xi[j]
            x[i] = (b[i]-sum)/A[i][i]
        check = 0    
        for x1,x2 in zip(xi, x):
            if abs(x1-x2) > e:
                check += 1
        if check == 0:
            Stop = True                 
    return x, count     
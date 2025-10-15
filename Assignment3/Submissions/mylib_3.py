import numpy as np

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

def row_swap(A, i, j):
    A[i], A[j] = A[j], A[i]
    return A

def Gauss_Seidel(A, b, e = 1e-6):
    n = len(A)
    x = [0 for i in range(n)]
    stop = 1
    count = 0
    while stop == True:
        sum = 0
        for i in range(n):
            xi = x[i]
            sum1, sum2 = 0, 0
            for j in range(i):
                sum1 += A[i][j]*x[j]
            for j in range(i+1, n):
                sum2 += A[i][j]*x[j]
            x[i] = (b[i] - sum1 - sum2)/A[i][i]
            delxi = (xi - x[i])**2
            sum += delxi
        d = np.sqrt(sum)
        if d < e:
            stop = False
        count += 1    
    return x, count

def is_symmetric(A):
    check = False
    if A == Transpose(A):
        check = True
    return check
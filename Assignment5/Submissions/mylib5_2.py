import numpy as np

def deflation(P, a):
    Q = P.copy()
    for i in range(1, len(P)):
        Q[i] = P[i] + Q[i-1]*a 
    return Q[:-1]

def function_list(L, x):
    sum = 0
    n = len(L)
    for i in range(n):
        sum += L[i]*x**(n-1-i)
    return sum    


def n_derivative(L, n):
    coeffs = L[:]
    for _ in range(n):
        m = len(coeffs)
        coeffs = [(m-1-j) * coeffs[j] for j in range(m-1)]
    return coeffs     



def Laguerre(P, guess=1, e = 1e-6):
    if function_list(P, guess) == 0:
     return guess
    n = len(P)-1
    x = guess
    while True:  
        # print(x)      
        G = function_list(n_derivative(P, 1), x)/function_list(P, x)
        H = G**2 - (function_list(n_derivative(P, 2), x)/function_list(P, x))
        if G > 0:
            if G+np.sqrt(abs((n-1)*(n*H-G**2))) == 0:
                a = 0
            else:    
                a = n/(G+np.sqrt(abs((n-1)*(n*H-G**2))))
        else:
            if G-np.sqrt(abs((n-1)*(n*H-G**2))) == 0:
                a = 0
            else:              
                a = n/(G-np.sqrt(abs((n-1)*(n*H-G**2))))
        x0, x = x, x - a

        if abs(x - x0)< e or abs(function_list(P, x)) < 1e-4:
            return x
        
def Laguerre_solve(P, guess):
    n = len(P)-1
    Q = P.copy()
    sols = []
    for i in range(n):
        sol = Laguerre(Q, guess[i])
        Q = deflation(Q, sol)
        sols.append(sol)
    return sols
 

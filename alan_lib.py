import numpy as np

class MyComplex():
    
    def __init__(self, real = 0, imag = 0.0):
        self.r = real
        self.i = imag

    def display_cmplx(self):
        print(f'{self.r} {self.i}j')

    def add_cmplx(self, c1, c2):
        self.r = c1.r + c2.r
        self.i = c1.i + c2.i
        return MyComplex(self)

    def sub_cmplx(self, c1, c2):
        self.r = c1.r - c2.r
        self.i = c1.i - c2.i
        return MyComplex(self)
    
    def mul_cmplx(self, c1, c2):
        self.r = c1.r*c2.r - c1.i*c2.i
        self.i = c1.i*c2.r + c1.r*c2.i
        return MyComplex(self)
    
    def mod_cmplx(self):
        return np.sqrt(self.r**2+self.i**2)
    

def iter_seq(c, seed,n = 10000):
    x = seed
    seq = [seed]
    for i in range(n-1):
        x = c*x*(1-x)
        seq.append(x)
    return seq

def LCG_rand_list(self, N):
        x = self.seed
        list = []
        for i in range(N):
            x = (self.a*x + self.c)%self.m
            list.append(x/self.m)
        return list  

class LinAlg:

    # ---------------- Row operations ----------------
    @staticmethod
    def row_swap(A, i, j):
        A[i], A[j] = A[j], A[i]
        return A

    @staticmethod
    def row_multiply(A, i, k):
        n = len(A[i])
        A[i] = [A[i][j]*k for j in range(n)]
        return A

    @staticmethod
    def row_add(A, i, j, k):
        n = len(A[i])
        A[i] = [A[i][c] + A[j][c]*k for c in range(n)]    
        return A

    # ---------------- Matrix helpers ----------------
    @staticmethod
    def transpose(A):
        n, m = len(A), len(A[0])
        return [[A[j][i] for j in range(n)] for i in range(m)]

    @staticmethod
    def augment(A, b):
        n = len(A)
        return [A[i] + [b[i]] for i in range(n)]

    @staticmethod
    def col_max(A, i):
        col_i = [abs(A[row][i]) for row in range(len(A))]
        return col_i.index(max(col_i))

    # ---------------- Solvers ----------------
    @staticmethod
    def gauss_jordan(A, b):
        n = len(A)
        index = LinAlg.col_max(A, 0)
        Aug = LinAlg.augment(A, b)
        LinAlg.row_swap(Aug, index, 0)
        for c in range(n):
            a = Aug[c][c]
            LinAlg.row_multiply(Aug, c, 1/a)
            for i in range(n):
                if i != c:    
                    a = Aug[i][c]
                    if a != 0:
                        LinAlg.row_add(Aug, i, c, -a)
        b = [Aug[i][n] for i in range(n)]                
        return {"Augmented Matrix": Aug, "Solution": b}  

    # ---------------- Products ----------------
    @staticmethod
    def dot(c, d):
        return sum(ci*di for ci, di in zip(c, d))

    @staticmethod
    def matmul(A, B):
        N, K, M = len(A), len(B), len(B[0])
        return [[round(sum(A[i][k] * B[k][j] for k in range(K)), 5) for j in range(M)] for i in range(N)]

    @staticmethod
    def matvec(A, b):
        m, n = len(A), len(A[0])
        return [sum(A[i][j] * b[j] for j in range(n)) for i in range(m)]

    # ---------------- LU decomposition ----------------
    @staticmethod
    def LU_decompose(A):
        n = len(A)
        L = [[0]*n for _ in range(n)]
        U = [[0]*n for _ in range(n)]
        for i in range(n):
            L[i][i] = 1
        for j in range(n):
            U[0][j] = A[0][j]
        for j in range(n):
            for i in range(1, j+1):
                s = sum(L[i][k]*U[k][j] for k in range(i))
                U[i][j] = A[i][j] - s
            for i in range(j, n):
                s = sum(L[i][k]*U[k][j] for k in range(j))
                L[i][j] = (A[i][j] - s)/U[j][j]   
        return {"L": L, "U": U}   

    @staticmethod
    def LU_solve(A, b):
        n = len(A)
        sol = LinAlg.LU_decompose(A)
        L, U = sol["L"], sol["U"]
        y = []
        for i in range(n):
            s = sum(L[i][k]*y[k] for k in range(i))
            y.append(b[i] - s)
        x = [0]*n
        for i in range(n-1, -1, -1):
            s = sum(U[i][k]*x[k] for k in range(i+1, n))
            x[i] = (y[i] - s)/U[i][i]
        return x

    # ---------------- Utility ----------------
    @staticmethod
    def vecfunc(g, x):
        return [f(x) for f in g]

    @staticmethod
    def matfunc(J, x):
        return [[f(x) for f in row] for row in J]

    @staticmethod
    def norm(x):
        return sum(i**2 for i in x)**0.5

    @staticmethod
    def Id(n):
        X = [[0]*n for _ in range(n)]
        for i in range(n):
            X[i][i] = 1
        return X

    # ---------------- Inverse ----------------
    @staticmethod
    def augment_inv(A):
        n = len(A)
        X = LinAlg.Id(n)
        return [A[i] + X[i] for i in range(n)]

    @staticmethod
    def invert(A):
        n = len(A)
        index = LinAlg.col_max(A, 0)
        Aug = LinAlg.augment_inv(A)
        LinAlg.row_swap(Aug, index, 0)
        for c in range(n):
            a = Aug[c][c]
            LinAlg.row_multiply(Aug, c, 1/a)
            for i in range(n):
                if i != c:    
                    a = Aug[i][c]
                    if a != 0:
                        LinAlg.row_add(Aug, i, c, -a)
        return [row[n:] for row in Aug]

class Linsolve:

    def bisection(f, guess, e = 1e-6):
        interval = guess
        count = 0
        while True:
            count += 1
            a, b = interval[0], interval[1]
            if f(a)*f(b) < 0:
                if abs(b - a) < e and (f(a),f(b)) < (e, e):
                    return (b+a)/2, count
                else:
                    c = (a+b)/2
                    if f(c)*f(a) < 0:
                        interval = [a, c]
                    else:
                        interval = [c, b]


    def regula_falsi(f, guess, e = 1e-6):
        interval = guess
        count = 0
        while True:
            count += 1
            a, b = interval[0], interval[1]
            c = b - (b-a)*f(b)/(f(b)-f(a))
            if f(a)*f(b) < 0:
                if abs(b - a) < e or abs(f(c)) < e:
                    return c, count
                else:  
                    if f(c)*f(a) < 0:
                        interval = [a, c]
                    else:
                        interval = [c, b]
            else:
                print('wrong bracket')
                break
                        

    def bracketing(f, interval, beta = 1):
        count = 0
        while True:
            count += 1
            a, b = interval[0], interval[1]
            if f(a)*f(b) <= 0:
                return interval, count
            else:
                diff = abs(f(b)) - abs(f(a))
                if diff > 0:
                    a -= beta*(b - a)
                    interval = [a, b]
                else:
                    b += beta*(b-a)
                    interval = [a, b]

    def newton_raphson(f, df, guess, e = 1e-6):
        x = guess
        count = 0
        while True:  
            count += 1      
            f_val = f(x) 
            df_val = df(x)
            x0, x = x, x - f_val/df_val 
            if abs(x - x0) < e and abs(f(x)) < e:
                    return x, count 
                            
    def fixed_point(f, g, guess, e = 1e-6):
        x = guess
        count = 0
        while True:  
            count += 1
            x0, x = x, g(x)
            if abs(x - x0) < e:
                    return x, count                              
            
class Integrate:  
    @staticmethod
    def Midpoint(f, a, b, n):
        h = (b-a)/n
        sum = 0
        for i in range(n):
            xn = a + (2*i+1)*h/2
            sum += h * f(xn)
        return sum

    @staticmethod
    def Trapezoid(f, a, b, n):
        h = (b-a)/n
        sum = f(a)
        for i in range(1, n):
            xn = a + i*h
            sum += 2*f(xn)

        return h*(sum+f(b))/2

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


def lcg_rng(n,x0,a = 1103515245,c = 12345,m = 32768 ,range=None):
    out=[]
    i=0
    tempx=x0
    while i<=n:
        temp=(a*tempx + c)%m
        tempx=temp
        out.append(tempx)
        i+=1
    if range==None:
        return out
    else:
        d=range[1]-range[0]
        templ=[i/m for i in out]
        out2=[range[0] + d*x for x in templ]
        return out2 

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
    def read_matrix(filename):
        with open(filename, 'r') as f:
            matrix = []
            for line in f:
                row = [float(num) for num in line.strip().split()]
                matrix.append(row)
        return matrix

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
    def Midpoint(f, a, b, n):
        h = (b-a)/n
        sum = 0
        for i in range(n):
            xn = a + (2*i+1)*h/2
            sum += h * f(xn)
        return sum

    def Trapezoid(f, a, b, n):
        h = (b-a)/n
        sum = f(a)
        for i in range(1, n):
            xn = a + i*h
            sum += 2*f(xn)

        return h*(sum+f(b))/2

    def Simpson(f, a, b, n):
            h = (b-a)/n
            sum = f(a)
            for i in range(1, n):
                xn = a + i*h
                if i%2 == 0:
                    sum += 2*f(xn)
                else:
                    sum += 4*f(xn)
                        

            return h*(sum+f(b))/3
    
    def Montecarlo(f, a, b, n):
        L = lcg_rng(n, x0=0.1, range=[a,b])
        h = (b-a)/n
        sum1, sum2 = 0, 0
        for xi in L:
            sum1 += f(xi)
            sum2 += f(xi)**2
        var = sum2/n - (sum1/n)**2
        return h*(sum1), var    
    
    def gaussian_quad(n, f, a, b):
        roots, weights = np.polynomial.legendre.leggauss(n)
        int = [float(weights[i])*float(f(((b-a)/2)*roots[i]+((b+a)/2))) for i in range(n)]
        return ((b-a)/2)*sum(int)
    
    def gaussian_quad_accur(f, val, accur, a, b):
        n = 1
        while True:
            int = Integrate.gaussian_quad(n, f, a, b)
            if abs(int-val)<=accur:
                return int, n
                break
            else:
                n+=1    

    def simpson_accur(f, a, b, val, accur):
        n = 2
        while True:
            int = Integrate.Simpson(f, a, b, n)
            if abs(int-val)<=accur:
                return int, n
                break
            else:
                n+=2                

    def Euler_Forward(f, range, y0, h = 0.1):
        x, y, x_max = range[0], y0, range[1]
        x_vals = [x]
        y_vals = [y]
        while x < x_max:
            y += h*f(y, x)
            x += h
            y_vals.append(y)
            x_vals.append(x)
        return x_vals, y_vals

    def Predictor_corrector(f, range, y0, h = 0.1):
        x, y, x_max = range[0], y0, range[1]
        x_vals = [x]
        y_vals = [y]
        while x < x_max:
            k1 = h*f(y, x)
            yp = y + k1
            k2 = h*f(yp, x+h)
            y += (k1+k2)/2
            x += h
            y_vals.append(y)
            x_vals.append(x)
        return x_vals, y_vals                

    # def X_dot(F, X, t):
    #     return np.array([f(X, t) for f in F])


    # def Predictor_corrector_list(F, range, X0, h = 0.1):
    #         t, X, t_max = range[0], X0, range[1]
    #         t_vals = [t]
    #         X_vals = [X]
    #         while t < t_max:
    #             k1 = h*X_dot(F, X, t)
    #             Xp = X + k1
    #             k2 = h*X_dot(F, Xp+h, t)
    #             X += (k1+k2)/2
    #             t += h
    #             X_vals.append(X)
    #             t_vals.append(t)
    #         return t_vals, X_vals                
    # t, x = Predictor_corrector(F, [0, 40], [1, 0])

    def rk4_step(x, y, x_dot, y_dot, h):
        k1 = x_dot(x, y)
        l1 = y_dot(x, y)

        k2 = x_dot(x + 0.5 * h * k1, y + 0.5 * h * l1)
        l2 = y_dot(x + 0.5 * h * k1, y + 0.5 * h * l1)

        k3 = x_dot(x + 0.5 * h * k2, y + 0.5 * h * l2)
        l3 = y_dot(x + 0.5 * h * k2, y + 0.5 * h * l2)

        k4 = x_dot(x + h * k3, y + h * l3)
        l4 = y_dot(x + h * k3, y + h * l3)

        K = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        L = (l1 + 2 * l2 + 2 * l3 + l4) / 6

        x += K * h
        y += L * h

        return x, y
    
    def Runge_kutta_2d(x_dot, y_dot, range, init, h = 0.1):
        x, y, t0, t_max = init[0], init[1], range[0], range[1]
        x_vals, y_vals = [x], [y]
        t_vals = np.arange(t0, t_max, h)
        for _ in t_vals[1:]:
            x, y = Integrate.rk4_step(x, y, x_dot, y_dot, h)
            x_vals.append(x)
            y_vals.append(y)
        return x_vals, y_vals, t_vals      

    def Runge_kutta(f, range, y0, h = 0.1):
            x, y, x_max = range[0], y0, range[1]
            x_vals = [x]
            y_vals = [y]
            while x < x_max:
                k1 = h*f(y, x)
                k2 = h*f(y + k1/2, x+h/2)
                k3 = h*f(y + k2/2, x+h/2)
                k4 = h*f(y + k3, x+h)
                y += (k1+2*k2+2*k3+k4)/6
                x += h
                y_vals.append(y)
                x_vals.append(x)
            return x_vals, y_vals

    def shooting(x_dot, y_dot, guess, boundary, range):#boundary = [start, end], range = [t0, tf]
        end = boundary[1]
        init = [boundary[0], guess]
        x, y, t = Integrate.Runge_kutta_2d(x_dot, y_dot, range, init)
        return x[-1]

    def BVP(x_dot, y_dot, guess, boundary, range):
        start, end, gl, gh = boundary[0], boundary[1], guess[0], guess[1]
        init = [start, gl]
        x1, y1, t1 = Integrate.Runge_kutta_2d(x_dot, y_dot, range, init)
        init = [start, gh]
        x2, y2, t2 = Integrate.Runge_kutta_2d(x_dot, y_dot, range, init)
        end_l, end_h = x1[-1], x2[-1]
        guess_new = gl + (gh - gl)*(end-end_l)/(end_h - end_l)
        init = [start, guess_new]
        x, y, t = Integrate.Runge_kutta_2d(x_dot, y_dot, range, init)
        return {'new_guess' : guess_new, 
        'overestimate' : (x2, y2, t2), 
        'underestimate': (x1, y1, t1),
        'solution':(x, y, t)}

    def forward_matrix(n, a):
            X = [[0]*n for _ in range(n)]
            for i in range(n):
                X[i][i] = 1 - 2*a
                if i < n-1:
                    X[i][i+1] = a
                if i > 0:
                    X[i][i-1] = a 
            return X
       
    def PDE_solve(u0, boundary, N, hx=0.05, ht=0.0005):
        start, end = boundary
        n = int(((end-start)/hx)+1)
        x_vals = [start + i*hx for i in range(n)]
        v = [u0(x) for x in x_vals]
        a = ht / hx**2
        A = Integrate.forward_matrix(n, a)
        soln = [v.copy()]

        for i in range(N):
            v[0], v[-1] = 0, 0           # enforce BC before
            v = LinAlg.matvec(A, v)      # matrix multiply
            v[0], v[-1] = 0, 0           # enforce BC after
            soln.append(v.copy())

        return soln, x_vals
        
class LinReg:
    def lagrange_interpolation(init, x):#init = [(x0, y0)]
        n = len(init)
        sum = 0
        for i in range(n):
            prod = 1
            for k in range(n):
                if k != i:
                    prod *= (x-init[k][0])/(init[i][0]-init[k][0])
            sum += prod*init[i][1]
        return sum

    def least_square_fit(init,s):
        n = len(init)
        S, Sx, Sy, Sxx, Sxy, Syy = 0, 0, 0, 0, 0, 0
        for i in range(n):
            h, xi, yi  = 1/s[i]**2, init[i][0], init[i][1]
            S += h
            Sx += h*xi
            Sxx += h*xi**2
            Sy += yi*h
            Sxy += xi*yi*h
            Syy += h*yi**2
        delta = S*Sxx - Sx**2
        a1, a2 = (Sxx*Sy - Sx*Sxy)/delta, (Sxy*S - Sx*Sy)/delta
        r2 = Sxy**2/(Sxx*Syy)
        return a1, a2, r2

    def ln_tuple(data, type):# type- 1 = log-log , 2 = x-log
        L = []
        if type == 1:
            for x in data:
                L.append((np.log(x[0]), np.log(x[1])))
        if type == 2:
            for x in data:
                L.append((x[0], np.log(x[1])))        
        return L            
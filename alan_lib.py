import numpy as np

# class MyComplex():
    
#     def __init__(self, real = 0, imag = 0.0):
#         self.r = real
#         self.i = imag

#     def display_cmplx(self):
#         print(f'{self.r} {self.i}j')

#     def add_cmplx(self, c1, c2):
#         self.r = c1.r + c2.r
#         self.i = c1.i + c2.i
#         return MyComplex(self)

#     def sub_cmplx(self, c1, c2):
#         self.r = c1.r - c2.r
#         self.i = c1.i - c2.i
#         return MyComplex(self)
    
#     def mul_cmplx(self, c1, c2):
#         self.r = c1.r*c2.r - c1.i*c2.i
#         self.i = c1.i*c2.r + c1.r*c2.i
#         return MyComplex(self)
    
#     def mod_cmplx(self):
#         return np.sqrt(self.r**2+self.i**2)
    
class RandSeq:
    def iter_seq(n, c=3.2 , seed = 0.1):  # n=number of terms, c=logistic parameter, seed=initial value
        x = seed
        seq = [seed]
        for i in range(n-1):
            x = c*x*(1-x)
            seq.append(x)
        return seq  # returns list of logistic sequence values


    def lcg_rng(n,seed  = 10,a = 1103515245,c = 12345,m = 32768 ,range=None):  # n=count, seed=start, a,c,m=LCG params, range=[min,max] or None
        out=[]
        i=0
        tempx=seed
        while i<n:
            temp=(a*tempx + c)%m
            tempx=temp
            out.append(tempx)
            i+=1
        if range==None:
            return out  # returns list of raw LCG integers in [0,m)
        else:
            d=range[1]-range[0]
            templ=[i/m for i in out]
            out2=[range[0] + d*x for x in templ]
            return out2  # returns list of scaled pseudo-random numbers in given range 

class LinAlg:

    # ---------------- Row operations ----------------
    def row_swap(A, i, j):  # A=matrix, i=row index, j=row index
        A[i], A[j] = A[j], A[i]
        return A  # returns matrix A with rows i and j swapped

    def row_multiply(A, i, k):  # A=matrix, i=row index, k=scalar multiplier
        n = len(A[i])
        A[i] = [A[i][j]*k for j in range(n)]
        return A  # returns matrix A with row i scaled

    def row_add(A, i, j, k):  # A=matrix, i=target row, j=source row, k=scalar multiple of row j
        n = len(A[i])
        A[i] = [A[i][c] + A[j][c]*k for c in range(n)]    
        return A  # returns matrix A after row i ← row i + k*row j

    # ---------------- Matrix helpers ----------------
    def Transpose(A):  # A=matrix
        n = len(A)
        L = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                L[i][j] = A[j][i]
        return L  # returns transpose of matrix A

    def augment(A, b):  # A=matrix, b=vector
        n = len(A)
        return [A[i] + [b[i]] for i in range(n)]  # returns augmented matrix [A|b]

    def col_max(A, i):  # A=matrix, i=column index
        col_i = [abs(A[row][i]) for row in range(len(A))]
        return col_i.index(max(col_i))  # returns row index of maximum absolute value in column i  

    # ---------------- Solvers ----------------
    def gauss_jordan(A, b):  # A=coefficient matrix, b=RHS vector
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
        return {"Augmented Matrix": Aug, "Solution": b}  # returns final augmented matrix and solution vector  
    
    def invert_gaussj(A):  # A=matrix to invert
        n = len(A)
        index = LinAlg.col_max(A, 0)
        X = LinAlg.Id(n)
        Aug = [A[i] + X[i] for i in range(n)]
        LinAlg.row_swap(Aug, index, 0)
        for c in range(n):
            a = Aug[c][c]
            LinAlg.row_multiply(Aug, c, 1/a)
            for i in range(n):
                if i != c:    
                    a = Aug[i][c]
                    if a != 0:
                        LinAlg.row_add(Aug, i, c, -a)
        return [row[n:] for row in Aug]  # returns inverse matrix of A    

    def Gauss_Seidel(A, b, e = 1e-6):  # A=matrix, b=RHS vector, e=error tolerance
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
        return x, count  # returns approximate solution vector and iteration count    

    def Cholesky_decomposition(A):  # A=symmetric positive definite matrix
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
        return L  # returns upper-triangular factor L such that A = (L^T)L  

    def Cholesky_solve(A, b):  # A=symmetric positive definite matrix, b=RHS vector
        n = len(A)
        U = LinAlg.Cholesky_decomposition(A)
        L = LinAlg.Transpose(U)
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
        return x  # returns solution vector x of Ax=b

    def Jacobi_iter(A, b, e = 1e-6, x = None):  # A=matrix, b=RHS vector, e=error tolerance, x=initial guess (unused)
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
        return x, count  # returns approximate solution vector and iteration count  

    # ---------------- Products ----------------
    def dot(c, d):  # c=vector, d=vector
        return sum(ci*di for ci, di in zip(c, d))  # returns dot product of c and d
    
    def matmul(A, B):  # A=matrix, B=matrix
        N, K, M = len(A), len(B), len(B[0])
        return [[round(sum(A[i][k] * B[k][j] for k in range(K)), 5) for j in range(M)] for i in range(N)]  # returns matrix product A*B

    def matvec(A, b):  # A=matrix, b=vector
        m, n = len(A), len(A[0])
        return [sum(A[i][j] * b[j] for j in range(n)) for i in range(m)]  # returns matrix-vector product A*b

    # ---------------- LU decomposition ----------------
    def LU_decompose(A):  # A=square matrix
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
        return {"L": L, "U": U}   # returns lower and upper triangular matrices L and U  

    def LU_solve(A, b):  # A=matrix, b=RHS vector
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
        return x  # returns solution vector x of Ax=b
    
    def LU_inverse(A):  # A=matrix to invert
        n = len(A)
        Ai = []
        for i in range(n):
            ei = [1 if j == i  else 0 for j in range(n)]
            Ai.append(LinAlg.LU_solve(A, ei))
        return LinAlg.Transpose(Ai)  # returns inverse of matrix A    

    # ---------------- Utility ----------------
    def read_matrix(filename):  # filename=path to text file containing matrix
        with open(filename, 'r') as f:
            matrix = []
            for line in f:
                row = [float(num) for num in line.strip().split()]
                matrix.append(row)
        return matrix  # returns matrix read from file

    def vecfunc(g, x):  # g=list of functions, x=input value
        return [f(x) for f in g]  # returns list of function values f(x) for each f in g

    def matfunc(J, x):  # J=matrix of functions, x=input value
        return [[f(x) for f in row] for row in J]  # returns matrix of function values evaluated at x

    def norm(x):  # x=vector
        return sum(i**2 for i in x)**0.5  # returns Euclidean norm of vector x

    def Id(n):  # n=dimension
        X = [[0]*n for _ in range(n)]
        for i in range(n):
            X[i][i] = 1
        return X  # returns n×n identity matrix


class Linsolve:

    # ---------------- Bracketing Methods ----------------
    def bisection(f, guess, e=1e-6):  # f=function, guess=[a,b] interval, e=error tolerance
        interval = guess; count = 0
        while True:
            count += 1
            a, b = interval
            if f(a)*f(b) < 0:
                if abs(b-a) < e and abs(f(a)) < e and abs(f(b)) < e:
                    return (a+b)/2, count  # returns root approximation and iteration count
                c = (a+b)/2
                interval = [a, c] if f(c)*f(a) < 0 else [c, b]
            else:
                print("Initial interval does not bracket a root")
                break

    def regula_falsi(f, guess, e=1e-6):  # f=function, guess=[a,b] interval, e=error tolerance
        interval = guess; count = 0
        while True:
            count += 1
            a, b = interval
            c = b - (b-a)*f(b)/(f(b)-f(a))
            if f(a)*f(b) < 0:
                if abs(b-a) < e or abs(f(c)) < e:
                    return c, count  # returns root approximation and iteration count
                interval = [a, c] if f(c)*f(a) < 0 else [c, b]
            else:
                print('wrong bracket')
                break

    def bracketing(f, interval, beta=1):  # f=function, interval=[a,b], beta=expansion factor
        count = 0
        while True:
            count += 1
            a, b = interval
            if f(a)*f(b) <= 0:
                return interval, count  # returns new bracketing interval and iteration count
            if abs(f(b)) - abs(f(a)) > 0:
                a -= beta*(b-a)
            else:
                b += beta*(b-a)
            interval = [a, b]

    # ---------------- Open Methods ----------------
    def newton_raphson(f, df, guess, e=1e-6):  # f=function, df=derivative, guess=initial x, e=error tolerance
        x = guess; count = 0
        while True:
            count += 1
            x0, x = x, x - f(x)/df(x)
            if abs(x-x0) < e and abs(f(x)) < e:
                return x, count  # returns root approximation and iteration count

    def fixed_point(f, g, guess, e=1e-6):  # f=original function (unused), g=iteration function, guess=initial x, e=error tolerance
        x = guess; count = 0
        while True:
            count += 1
            x0, x = x, g(x)
            if abs(x-x0) < e:
                return x, count  # returns fixed point and iteration count

    # ---------------- Multivariate Methods ----------------
    def vector(g, x):  # g=list of functions, x=vector
        return [f(x) for f in g]  # returns list of function values at x

    def multivar_fixed_point(g, guess, e=1e-6):  # g=list of vector functions, guess=initial vector, e=error tolerance
        norm = LinAlg.norm
        x = guess; count = 0
        while True:
            count += 1
            x0 = np.array(x)
            x  = np.array(Linsolve.vector(g, x))
            if norm(x - x0)/norm(x) < e:
                return x, count  # returns fixed point vector and iteration count

    def Jacobian(J, x):  # J=Jacobian as matrix of functions, x=vector
        n = len(J)
        return [[f(x) for f in J[i]] for i in range(n)]  # returns numeric Jacobian matrix at x

    def multivar_newton_raphson(f, J, guess, e=1e-6):  # f=list of functions, J=Jacobian functions, guess=initial vector, e=error tolerance
        norm = LinAlg.norm
        x = guess; count = 0
        while True:
            count += 1
            f_val = Linsolve.vector(f, x)
            Jin   = LinAlg.invert_gaussj(Linsolve.Jacobian(J, x))
            y     = np.array(LinAlg.matvec(Jin, f_val))
            x     = np.array(x)
            x0, x = x, x - y
            if norm(x - x0)/norm(x) < e:
                return x, count  # returns root vector and iteration count

    # ---------------- Polynomial Tools ----------------
    def deflation(P, a):  # P=polynomial coefficients, a=root to deflate at
        Q = P.copy()
        for i in range(1, len(P)):
            Q[i] = P[i] + Q[i-1]*a
        return Q[:-1]  # returns deflated polynomial coefficients

    def function_list(L, x):  # L=coefficients list, x=value
        total = 0; n = len(L)
        for i in range(n):
            total += L[i]*x**(n-1-i)
        return total  # returns polynomial value at x

    def n_derivative(L, n):  # L=coefficients list, n=order of derivative
        coeffs = L[:]
        for _ in range(n):
            m = len(coeffs)
            coeffs = [(m-1-j)*coeffs[j] for j in range(m-1)]
        return coeffs  # returns coefficients of n-th derivative

    # ---------------- Laguerre Root Solver ----------------
    def Laguerre(P, guess=1, e=1e-6):  # P=polynomial coefficients, guess=initial root guess, e=error tolerance
        FL = Linsolve.function_list
        ND = Linsolve.n_derivative
        if FL(P, guess) == 0:
            return guess  # returns exact root if guess is already a root
        n = len(P)-1; x = guess
        while True:
            Px = FL(P, x)
            P1 = FL(ND(P, 1), x)
            P2 = FL(ND(P, 2), x)
            G  = P1/Px
            H  = G**2 - (P2/Px)
            rad = np.sqrt(abs((n-1)*(n*H - G**2)))
            denom = G + rad if G > 0 else G - rad
            a = 0 if denom == 0 else n/denom
            x0, x = x, x - a
            if abs(x-x0) < e or abs(FL(P, x)) < 1e-4:
                return x  # returns one root of polynomial P

    def Laguerre_solve(P, guess):  # P=polynomial coefficients, guess=list of initial guesses for each root
        n = len(P)-1; Q = P.copy(); sols = []
        for i in range(n):
            r = Linsolve.Laguerre(Q, guess[i])
            Q = Linsolve.deflation(Q, r)
            sols.append(r)
        return sols  # returns list of all roots of polynomial P

 


class Integrate:  
    def Midpoint(f, a, b, n):  # f=function, a=start, b=end, n=number of subintervals
        h = (b-a)/n
        sum = 0
        for i in range(n):
            xn = a + (2*i+1)*h/2
            sum += h * f(xn)
        return sum  # returns Midpoint-rule approximation of integral

    def Trapezoid(f, a, b, n):  # f=function, a=start, b=end, n=number of subintervals
        h = (b-a)/n
        sum = f(a)
        for i in range(1, n):
            xn = a + i*h
            sum += 2*f(xn)

        return h*(sum+f(b))/2  # returns Trapezoidal-rule approximation of integral

    def Simpson(f, a, b, n):  # f=function, a=start, b=end, n=even number of subintervals
            if n%2 != 0:
                print('n should be even')
                return None
            h = (b-a)/n
            sum = f(a)
            for i in range(1, n):
                xn = a + i*h
                if i%2 == 0:
                    sum += 2*f(xn)
                else:
                    sum += 4*f(xn)
            return h*(sum+f(b))/3  # returns Simpson-rule approximation of integral
    
    def Montecarlo(f, a, b, n):  # f=function, a=start, b=end, n=number of random samples
        L = RandSeq.lcg_rng(n, seed=0.1, range=[a,b])
        h = (b-a)/n
        sum1, sum2 = 0, 0
        for xi in L:
            sum1 += f(xi)
            sum2 += f(xi)**2
        var = sum2/n - (sum1/n)**2
        return h*(sum1), var  # returns Monte Carlo integral estimate and variance    
    
    def gaussian_quad(n, f, a, b):# n=number of nodes, f=function, a=start, b=end
        roots, weights = np.polynomial.legendre.leggauss(n)
        int = [float(weights[i])*float(f(((b-a)/2)*roots[i]+((b+a)/2))) for i in range(n)]
        return ((b-a)/2)*sum(int)#Integral result
    
    def gaussian_quad_accur(f, val, accur, a, b):  # f=function, val=target value, accur=desired accuracy, a=start, b=end
        n = 1
        while True:
            int = Integrate.gaussian_quad(n, f, a, b)
            if abs(int-val)<=accur:
                return int, n  # returns integral approximation and nodes used
            else:
                n+=1    

    def simpson_accur(f, a, b, val, accur):  # f=function, a=start, b=end, val=target value, accur=desired accuracy
        n = 2
        while True:
            int = Integrate.Simpson(f, a, b, n)
            if abs(int-val)<=accur:
                return int, n  # returns Simpson integral value and number of subintervals
                break
            else:
                n+=2                

    def Euler_Forward(f, range, y0, h = 0.1):  # f=ODE f(y,x), range=[x0,xmax], y0=initial value, h=step size
        x, y, x_max = range[0], y0, range[1]
        x_vals = [x]
        y_vals = [y]
        while x < x_max:
            y += h*f(y, x)
            x += h
            y_vals.append(y)
            x_vals.append(x)
        return x_vals, y_vals  # returns lists of x and y values

    def Predictor_corrector(f, range, y0, h = 0.1):  # f=ODE f(y,x), range=[x0,xmax], y0=initial value, h=step size
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
        return x_vals, y_vals  # returns lists of x and y values                

    def rk4_step(x, y, x_dot, y_dot, h):  # x=current x, y=current y, x_dot=dx/dt, y_dot=dy/dt, h=step size
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

        return x, y  # returns updated values of x and y
    
    def Runge_kutta_2d(x_dot, y_dot, range, init, h = 0.1):  # x_dot=dx/dt, y_dot=dy/dt, range=[t0,tf], init=[x0,y0], h=step size
        x, y, t0, t_max = init[0], init[1], range[0], range[1]
        x_vals, y_vals = [x], [y]
        t_vals = np.arange(t0, t_max, h)
        for _ in t_vals[1:]:
            x, y = Integrate.rk4_step(x, y, x_dot, y_dot, h)
            x_vals.append(x)
            y_vals.append(y)
        return x_vals, y_vals, t_vals  # returns lists of x(t), y(t), and times      

    def Runge_kutta(f, range, y0, h = 0.1):  # f=ODE f(y,x), range=[x0,xmax], y0=initial value, h=step size
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
            return x_vals, y_vals  # returns lists of x and y values
    
    def Runge_kutta_vectorised(F, init, t_range, h = 0.1):  # F=vector field F(X), init=initial vector, t_range=[t0,tf], h=step size
                X= np.array(init)
                t_vals = np.arange(t_range[0], t_range[1]+h, h)
                X_vals = [X.copy()]
                for _ in t_vals[1:]:
                    k1 = h*F(X)
                    k2 = h*F(X + k1/2)
                    k3 = h*F(X + k2/2)
                    k4 = h*F(X + k3)
                    X = X + (k1+2*k2+2*k3+k4)/6
                    X_vals.append(X.copy())
                return t_vals, np.array(X_vals)  # returns time grid and array of solution vectors    

    def shooting(x_dot, y_dot, guess, boundary, range):  # x_dot=dx/dt, y_dot=d^2x/dt^2 or similar, guess=initial slope, boundary=[start,end], range=[t0,tf]
        end = boundary[1]
        init = [boundary[0], guess]
        x, y, t = Integrate.Runge_kutta_2d(x_dot, y_dot, range, init)
        return x[-1]  # returns solution value at final point for given guess

    def BVP(x_dot, y_dot, guess, boundary, range):  # x_dot=dx/dt, y_dot=d^2x/dt^2, guess=[low,high] initial slopes, boundary=[start,end], range=[t0,tf]
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
        'solution':(x, y, t)}  # returns dictionary with new guess and solution curves

    def forward_matrix(n, a):  # n=number of grid points, a=dt/dx^2 parameter
            X = [[0]*n for _ in range(n)]
            for i in range(n):
                X[i][i] = 1 - 2*a
                if i < n-1:
                    X[i][i+1] = a
                if i > 0:
                    X[i][i-1] = a 
            return X  # returns tridiagonal forward-time matrix for heat equation
       
    def PDE_solve(u0, boundary, N, hx=0.05, ht=0.0005):  # u0=initial profile function, boundary=[x_start,x_end], N=time steps, hx=space step, ht=time step
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

        return soln, x_vals  # returns list of solution profiles and spatial grid
        
class LinReg:
    def lagrange_interpolation(init, eval):  # init=list of (x,y) points, eval=x value to evaluate at
        n = len(init)
        sum = 0
        for i in range(n):
            prod = 1
            for k in range(n):
                if k != i:
                    prod *= (eval-init[k][0])/(init[i][0]-init[k][0])
            sum += prod*init[i][1]
        return sum  # returns interpolated y value at eval

    def least_square_fit(init, s):  # init=list of (x,y) points, s=list of standard deviations for y
        n = len(init)
        S, Sx, Sy, Sxx, Sxy, Syy = 0, 0, 0, 0, 0, 0
        
        for i in range(n):
            h, xi, yi = 1/s[i]**2, init[i][0], init[i][1]
            S += h
            Sx += h*xi
            Sxx += h*xi**2
            Sy += h*yi
            Sxy += h*xi*yi
            Syy += h*yi**2

        delta = S*Sxx - Sx**2

        a1 = (Sxx*Sy - Sx*Sxy)/delta
        a2 = (Sxy*S - Sx*Sy)/delta

        sigma_a1 = (Sxx/delta)**0.5
        sigma_a2 = (S/delta)**0.5

        r2 = Sxy**2/(Sxx*Syy)

        return a1, a2, sigma_a1, sigma_a2, r2  # returns intercept, slope, their errors, and r^2

    def ln_tuple(data, type):   # data=list of (x,y), type=1 log-log, 2 x-log, 3 log-x
        L = []
        if type == 1:
            for x in data:
                L.append((np.log(x[0]), np.log(x[1])))
        elif type == 2:
            for x in data:
                L.append((x[0], np.log(x[1])))
        elif type == 3:
            for x in data:
                L.append((np.log(x[0]), x[1]))
        return L  # returns transformed list of (X,Y) tuples for fitting               
    

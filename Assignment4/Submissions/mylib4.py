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
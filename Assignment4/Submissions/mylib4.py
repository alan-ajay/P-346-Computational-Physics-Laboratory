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
        if f(a)*f(b) < 0:
            if abs(b - a) < e and (f(a),f(b)) < (e, e):
                return (b+a)/2, count
            else:  
                c = b - (b-a)*f(b)/(f(b)-f(a))
                if f(c)*f(a) < 0:
                    interval = [a, c]
                else:
                    interval = [c, b]

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
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
    
class random():

    def __init__(self, seed, a = 1103515245, c = 12345, m = 32768):
        self.a = a
        self.c = c
        self.m = m   
        self.seed = seed

    def rand_list(self):
        x = self.seed
        list = []
        for i in range(self.N):
            x = (self.a*x + self.c)%self.m
            list.append(x/self.m)
        return list  


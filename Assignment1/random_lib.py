class random():

    def __init__(self, seed = 10, a = 1103515245, c = 12345, m = 32768):
        self.a = a
        self.c = c
        self.m = m   
        self.seed = seed

    def LCG_rand_list(self, N):
        x = self.seed
        list = []
        for i in range(N):
            x = (self.a*x + self.c)%self.m
            list.append(x/self.m)
        return list  
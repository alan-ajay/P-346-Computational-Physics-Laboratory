
def LCG_rand_list(self, N):
        x = self.seed
        list = []
        for i in range(N):
            x = (self.a*x + self.c)%self.m
            list.append(x/self.m)
        return list  


def iter_seq(c, seed,n = 10000):
    x = seed
    seq = [seed]
    for i in range(n-1):
        x = c*x*(1-x)
        seq.append(x)
    return seq
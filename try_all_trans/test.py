import numpy as np
import torch

class Test0:
    def __init__(self, config:dict):
        self.a = config['a']
        self.b = config['b']
        self.c = config['c']

    def print(self):
        print(self.a, self.b, self.c)
    
    def add(self, x, y):
        return x + y
    
    def sub(self, x, y):
        self.print()
        ab = self.add(x, y)
        out = ab - self.c
        return out
class Test1(Test0):
    def __init__(self, config:dict):
        super().__init__(config)
        self.d = config['d']
    
    
    def sub(self, x, y):
        self.print()
        ab = self.add(x, y)
        out = ab - self.c*self.d
        print(f"a+b - self.c*self.d")
        return out
config = {'a':1, 'b':2, 'c':3, 'd':4}
t0 = Test0(config)
t0.sub(1, 2)
t1 = Test1(config)  
t1.sub(1, 2)
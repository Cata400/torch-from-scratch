import math
import numpy as np
import matplotlib.pyplot as plt
import random


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        
        self.grad = 0.0
        self._backward = lambda: None
        
        self._prev = set(_children)
        self._op = _op
        self.label = label
        
        
    def __repr__(self):
        return f"Value(data={self.data})"
    
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            
        out._backward = _backward
        return out
    
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
    
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            
        out._backward = _backward
        
        return out


    def __pow__(self, other):
        assert isinstance(other, (int, float)), "The exponent should be an int / float"
        out = Value(self.data**other, (self, ), f'**{other}')
        
        def _backward():
            self.grad += other * self.data**(other - 1) * out.grad
            
        out._backward = _backward
        
        return out
    
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            print(out.data, out.grad)
            self.grad += out.data * out.grad
            
        out._backward = _backward
        
        return out
    
    
    def tanh(self):
        n = self.data
        t = (math.exp(2 * n) - 1) / (math.exp(2 * n) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
            
        out._backward = _backward
        
        return out
    
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
                
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
            
            
    def __radd__(self, other):
        return self + other
    
    
    def __neg__(self):
        return (-1) * self
    
    
    def __sub__(self, other):
        return self + (-other)
    
    
    def __rsub__(self, other):
        return other + (-self)
    
    
    def __rmul__(self, other): 
        return self * other
    
    
    def __truediv__(self, other):
        return self * other**-1
    
    
    def __rtruediv__(self, other):
        return other * self**-1
    
    
if __name__ == '__main__':
    from utils import draw_dot
    
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    b = Value(6.881375870195432, label='b')

    x1w1 = x1 * w1; x1w1.label = 'x1 * w1'
    x2w2 = x2 * w2; x2w2.label = 'x2 * w2'

    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1 * w1 + x2 * w2'
    n = x1w1x2w2 + b; n.label = 'n'

    o = n.tanh(); o.label = 'o'

    o.backward()
    
    graph = draw_dot(o)
    graph.render(directory='./graphs', view=True)
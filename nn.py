import math
import numpy as np
import matplotlib.pyplot as plt
import random

from engine import Value

class Neuron:
    def __init__(self, in_features):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(in_features)]
        self.b = Value(random.uniform(-1, 1))
        
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
    
class Layer:
    def __init__(self, in_features, out_features):
        self.neurons = [Neuron(in_features) for _ in range(out_features)]
        
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        
        
class MLP:
    def __init__(self, in_features, out_features_list):
        sizes = [in_features] + out_features_list
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(out_features_list))]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    
if __name__ == '__main__':
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]

    ys = [1.0, -1.0, -1.0, 1.0]
    
    model = MLP(3, [4, 4, 1])
    
    for k in range(100):
        y_pred = [model(x) for x in xs]
        
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, y_pred)) / len(ys) # MSE
        
        for p in model.parameters():
            p.grad = 0
        
        loss.backward()
        
        for p in model.parameters():
            p.data -= 0.05 * p.grad
            
        print(k, loss.data)
        
    print(y_pred)
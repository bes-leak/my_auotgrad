from .tensor import Tensor
import numpy as np


class Linear:
    def __init__(self, ninput, noutput):
        self.ninput = ninput
        self.noutput = noutput
        self.w = Tensor(np.random.uniform(-1, 1, (ninput, noutput)))
        self.b = Tensor(np.random.uniform(-1, 1, noutput))

    def __call__(self, X):
        out = (X @ self.w + self.b).tanh()
        return out

    def parameters(self):
        return [self.w, self.b]



class MLP:
    def __init__(self, ninput, noutput):
        self.layers = [Linear(nin, nout) for nin, nout in zip([ninput] + noutput[:-1], noutput)]

    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
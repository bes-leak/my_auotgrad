import random
from autograd import Value


class Neuron:
    def __init__(self, ninput):
        self.w = [Value(random.uniform(-1.0, 1.0)) for _ in range(ninput)]
        self.b = Value(random.uniform(-1.0, 1.0))
        self.n = ninput

    def __call__(self, x):
        act = self.b

        if len(self.w) != len(x):
            raise ValueError("size of x is not equal to ninput")

        for i in range(self.n):
            act += self.w[i] * x[i]

        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class ScalarLayer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
        self.nout = nout

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class ScalarMLP:
    def __init__(self, nin, nout):
        sizes = [nin] + nout
        self.layers = [ScalarLayer(sizes[i], sizes[i+1]) for i in range(len(nout))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
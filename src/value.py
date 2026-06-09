from math import exp, log
class Value:
    def __init__(self, data, children=()):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data  * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other))

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad -= 1.0 * out.grad

        out._backward = _backward
        return out

    def __pow__(self, n):
        assert isinstance(n, (int, float))
        out = Value(self.data** n, (self,))
        def _backward():
            self.grad += n * self.data ** (n - 1) * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        return self * other ** -1

    def exp(self):
        out = Value(exp(self.data), (self,))

        def _backward():
            self.grad += exp(self.data) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        value = self.data if self.data > 0 else 0
        out = Value(value, (self,))
        def _backward():
            self.grad += (out.grad if self.data > 0 else 0.0)

        out._backward = _backward
        return out

    def log(self):
        out = Value(log(self.data), (self,))
        def _backward():
            self.grad += 1 / self.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


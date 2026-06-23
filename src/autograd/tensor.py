import numpy as np

class Tensor:
    def __init__(self, data, children=()):
        data = np.asarray(data, dtype=np.float32)
        self.data = data
        self._prev = children
        self.grad = np.zeros_like(data)
        self._backward = lambda: None

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other))


        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward

        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            self.grad += self._unbroadcast(out.grad)
            other.grad += other._unbroadcast(out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))

        def _backward():
            self.grad += self._unbroadcast(other.data * out.grad)
            other.grad += other._unbroadcast(self.data * out.grad)

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other * self


    def tanh(self):
        tanh = np.tanh(self.data)
        out = Tensor(tanh, (self,))

        def _backward():
            self.grad += (np.ones_like(tanh) - tanh**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        relu = np.maximum(0, self.data)
        out = Tensor(relu, (self,))

        def _backward():
            self.grad += (np.zeros_like(relu) + (relu > 0)) * out.grad

        out._backward = _backward
        return out

    def sum(self):
        sum = np.sum(self.data)
        out = Tensor(sum, (self,))
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def mean(self):
        mean = np.mean(self.data)
        out = Tensor(mean, (self,))

        def _backward():
            self.grad += np.ones_like(self.data) / self.data.size * out.grad

        out._backward = _backward
        return out


    def _unbroadcast(self, grad):

        sums = grad
        while sums.ndim > self.data.ndim:
            sums = np.sum(sums, axis=0)

        for dim in range(sums.ndim):
            if sums.shape[dim] != self.data.shape[dim]:
                sums = np.sum(sums, axis=dim, keepdims=True)

        return sums


    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)

        for v in reversed(topo):
            v._backward()


    def to_numpy(self):
        return np.asarray(self.data)



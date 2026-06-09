from value import Value
from math import exp

a = Value(3.0)
b = Value(exp(1))
n = Value(2.0)
f = b.log() * b         # f = a·b + a
f.backward()
print(f.data)
print(a.grad, b.grad, n.grad)


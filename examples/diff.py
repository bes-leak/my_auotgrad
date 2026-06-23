from autograd import Value

w1 = Value(2.0)
w2 = Value(0.0)


for i in range(100):
    f = (w1 + 2 * w2 - 5) ** 2

    w1.grad = 0.0
    w2.grad = 0.0

    f.backward()
    w1.data -= 0.05 * w1.grad
    w2.grad -= 0.05 * w2.grad

    print(f.data)
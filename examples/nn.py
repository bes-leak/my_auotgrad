import numpy as np
from autograd import MLP, Tensor

x_size = (100, 4)

X = np.random.uniform(low=-10, high=10, size=x_size)
y = np.sin(np.sum(X, axis=1)) + np.random.normal(loc=0, scale=2, size=x_size[0])

Xt = Tensor(X)
yt = Tensor(y)

mlp = MLP(x_size[1], [8, 16, 32, 16, 8, 1])
lr = 1e-2

for epoch in range(200):
    ypred = mlp(Xt)

    for parameter in mlp.parameters():
        parameter.zero_grad()

    loss = ((yt - ypred) * (yt - ypred)).mean()
    loss.backward()

    for parameter in mlp.parameters():
        parameter.data -= lr * parameter.grad

    print(epoch, loss.data)



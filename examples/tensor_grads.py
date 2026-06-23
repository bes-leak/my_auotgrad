from autograd import Tensor
import numpy as np

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
tensor = Tensor(matrix)
res = tensor.relu().mean()

res.backward()
print(tensor.grad)

eps = 1e-3
f_0 = res.to_numpy()
num_diff = np.zeros_like(matrix)


for i in range(3):
    for j in range(3):
        matrix[i, j] += eps
        np_res = np.maximum(0, matrix)
        np_res = np.mean(np_res)
        num_diff[i, j] = (np_res - f_0) / (eps)
        matrix[i, j] -= eps



print(num_diff)

print(np.allclose(tensor.grad, num_diff, atol=1e-3))

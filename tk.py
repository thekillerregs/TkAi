import numpy as np
import torch

# Matrix Multiplication

# Numpy
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
C = np.random.randn(3, 7)

print(np.round(A @ B, 2))
print(' ')
print(np.round(C.T @ A, 2))

# PyTorch
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C1 = torch.randn(4, 7)
#C2 = torch.tensor(C1, dtype=torch.float)

print(np.round(A @ B, 2))

import numpy as np
import torch

# Transposing matrixes

# Numpy
nM = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(nM), print(' '), print(nM.T), print(' '), print(nM.T.T)

# PyTorch
tv = torch.tensor([[1, 2, 3, 4]])
print(tv), print(' '), print(tv.T), print(' '), print(tv.T.T)


import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Numpy

v = np.array([1, 40, 2, -3])

minval = np.min(v)
maxval = np.max(v)

print('Min, max: %g, %g' % (minval, maxval))

minidx = np.argmin(v)
maxidx = np.argmax(v)

print('Min, max indices: %g,%g' % (minidx, maxidx))

# Martix

M = np.array([[0, 1, 10], [20, 8, 5]])

print(M, ' ')

minvals1 = np.min(M)  # Whole matrix
minvals2 = np.min(M, axis=0)  # Each column (across rows)
minvals3 = np.min(M, axis=1)  # Each row ( across columns)

print(minvals1)
print(minvals2)
print(minvals3)

# Pytorch

v = torch.tensor([1, 40, 2, -3])

minval = torch.min(v)
maxval = torch.max(v)

print('Min, max: %g, %g' % (minval, maxval))

minidx = torch.argmin(v)
maxidx = torch.argmax(v)

print('Min, max indices: %g, %g' % (minidx, maxidx))

M = torch.tensor([[0, 1, 10], [20, 8, 5]])
print(M), print(' ')

min1 = torch.min(M)

min2 = torch.min(M, axis=0)
min3 = torch.min(M, axis=1)

print(min2.values)
print(min2.indices)

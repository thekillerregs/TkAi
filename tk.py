import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Entropy & Cross-Entropy

p = .25

H = - (p * np.log(p))
print('Wrong Entropy: ' + str(H))

x = [.25, .75]

H = 0

for p in x:
    H += -(p * np.log(p))

print('Correct Entropy: ' + str(H))

p = .25

# Binary Entropy
H = -(p * np.log(p) + (1 - p) * np.log(1 - p))
print('Binary Entropy: ' + str(H))

# Cross-Entropy
p = [1, 0]  # sum=1
q = [.25, .75]  # sum=1

H = 0

for i in range(len(p)):
    H -= p[i] * np.log(q[i])

print('Cross-Entropy: ' + str(H))

# Binary Cross-Entropy
H = -(p[0] * np.log(q[0]) + p[1] * np.log(q[1]))

print('Binary Cross-Entropy: ' + str(H))

# Simplification
H = -np.log(q[0])
print('Manually Simplified:' + str(H))

# Pytorch
p_tensor = torch.Tensor(p)
q_tensor = torch.Tensor(q)

# P = Category Labels
# Q = Model Probability
x = F.binary_cross_entropy(q_tensor, p_tensor)

print(x)

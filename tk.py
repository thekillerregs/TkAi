import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Softmax

# Numpy
z = [1, 2, 3]

num = np.exp(z)
den = np.sum(np.exp(z))
sigma = num / den

print(sigma)
print(np.sum(sigma))
print(' ')

z = np.random.randint(-5, high=15, size=25)
print(z)

num = np.exp(z)
den = np.sum(np.exp(z))
sigma = num / den

plt.plot(z, sigma, 'ko')
plt.xlabel('Original number (z)')
plt.ylabel('Softmaxified $\sigma$')

plt.title('$\sum\sigma$ = %g' % np.sum(sigma))
plt.show()

# PyTorch

z = [1, 2, 3]

softfun = nn.Softmax(dim=0)

sigmaT = softfun(torch.Tensor(z))

print(sigmaT)

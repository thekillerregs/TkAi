import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

#Seeding

np.random.randn(5)

#Old method
np.random.seed(17)

print(np.random.randn(5))
print(np.random.randn(5))

#New method
randseed1 = np.random.RandomState(17)
randseed2 = np.random.RandomState(20210530)

print(randseed1.randn(5))
print(randseed2.randn(5))
print(randseed1.randn(5))
print(randseed2.randn(5))
print(np.random.randn(5))

#Pytorch
torch.randn(5)

torch.manual_seed(17)

print(torch.randn(5))

print(np.random.randn(5))


import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Mean and Variance

x = [1, 2, 4, 6, 5, 4, 0]

n = len(x)

mean1 = np.mean(x)
mean2 = np.sum(x) / n

print(mean1, mean2)

# Degree of Freedom!
# ddof default is 0, so in the formula below it would be the equivalent of (n-0) on that parameter...
# 0 is biased, 1 is unbiased
var1 = np.var(x, ddof=1)
var2 = (1 / (n - 1)) * np.sum((x - mean1) ** 2)

print(var1, var2)

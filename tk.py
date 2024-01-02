import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# T-test
n1 = 30
n2 = 40
mu1 = 1
mu2 = 2

data1 = mu1 + np.random.randn(n1)
data2 = mu2 + np.random.randn(n2)

plt.plot(np.zeros(n1), data1, 'ro', markerfacecolor='w', markersize=14)
plt.plot(np.ones(n2), data2, 'bs', markerfacecolor='w', markersize=14)
plt.xlim([-1, 2])
plt.xticks([0, 1], labels=['Group 1', 'Group 2'])

plt.show()

# t-testing
# p below .05 for significance
t, p = stats.ttest_ind(data1, data2)
print('t-value:', t)
print('p-value:', p)

plt.plot(0 + np.random.randn(n1) / 15, data1, 'ro', markerfacecolor='w', markersize=14)
plt.plot(1 + np.random.randn(n2) / 15, data2, 'bs', markerfacecolor='w', markersize=14)
plt.xlim([-1, 2])
plt.xticks([0, 1], labels=['Group 1', 'Group 2'])

plt.title(f't = {t:.2f}, p ={p:.3f}')
plt.show()
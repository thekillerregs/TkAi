import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def resource_path(filename: str) -> str:
    """Return the absolute path to a file in the 'resources' folder."""
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Parent directory of the Python folder
    return os.path.join(base_dir, 'resources', filename)


# Importing data
dataset = pd.read_csv(resource_path('Position_Salaries.csv'))

# Creating datasets
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1]

# Splitting the datasets into training and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Linear regression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Polynomial regression
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Visualizing
x_grid = np.arange(np.min(x), np.max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x, y, color='blue')
plt.plot(x, lin_reg.predict(x), color='red', label='Linear Regression')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.transform(x_grid)), color='green', label='Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Truth or Bluff (Linear Regression)')
plt.legend()
plt.show()

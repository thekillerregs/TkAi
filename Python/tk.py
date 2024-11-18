import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor


def resource_path(filename: str) -> str:
    """Return the absolute path to a file in the 'resources' folder."""
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Parent directory of the Python folder
    return os.path.join(base_dir, 'resources', filename)


# Importing data
dataset = pd.read_csv(resource_path('Position_Salaries.csv'))

# Creating datasets
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Decision Tree Regressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)

# Predicting Salary
pred_y = regressor.predict([[6.5]])

print(pred_y)

# Visualizing
x_grid = np.arange(min(x[:, 0]), max(x[:, 0]), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

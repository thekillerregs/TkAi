import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def resource_path(filename: str) -> str:
    """Return the absolute path to a file in the 'resources' folder."""
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Parent directory of the Python folder
    return os.path.join(base_dir, 'resources', filename)


# Importing data
dataset = pd.read_csv(resource_path('Model_Selection.csv'))

# Creating datasets
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Multiple Linear Regression
lin_regressor = LinearRegression()
lin_regressor.fit(x_train, y_train)

# Polynomial Regression
poly_reg = PolynomialFeatures(degree=4)
poly_reg.fit(x_train, y_train)

# Support Vector Regression
svr_train_y = y_train.reshape(len(y_train), 1)
svr_test_y = y_test.reshape(len(y_test), 1)

sc_x = StandardScaler()
sc_y = StandardScaler()
svr_x_train = sc_x.fit_transform(x_train)
svr_y_train = sc_y.fit_transform(svr_train_y)

svr_reg = SVR(kernel='rbf')
svr_reg.fit(svr_x_train, svr_y_train)

# Decision Tree Regression
tree_reg = DecisionTreeRegressor(random_state=0)
tree_reg.fit(x_train, y_train)

# Random Forest Regression
rf_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
rf_regressor.fit(x_train, y_train)

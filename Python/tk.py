import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def resource_path(filename: str) -> str:
    """Return the absolute path to a file in the 'resources' folder."""
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Parent directory of the Python folder
    return os.path.join(base_dir, 'resources', filename)


# Importing data
dataset = pd.read_csv(resource_path('Position_Salaries.csv'))

# Creating datasets
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Reshaping Y
y = y.reshape(len(y), 1)

# Feature Scaling
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

print(x, y)

# Support Vector Regression (SVR)
regressor = SVR(kernel='rbf')
regressor.fit(x, y)

# Predicting Salary
pred_y = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1, 1))

# Visualizing SVR curve
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x).reshape(-1, 1)), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

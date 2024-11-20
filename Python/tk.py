import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap


def resource_path(filename: str) -> str:
    """Return the absolute path to a file in the 'resources' folder."""
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Parent directory of the Python folder
    return os.path.join(base_dir, 'resources', filename)


# Importing data
dataset = pd.read_csv(resource_path('Social_Network_Ads.csv'))

# Creating datasets
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Logistic Regression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

# Predictions
spec_y_pred = classifier.predict(sc.transform([[30, 87000]]))

print(spec_y_pred)

y_pred = classifier.predict(x_test)

y_disp = np.concatenate((y_test.reshape(len(y_test), 1), y_pred.reshape(len(y_pred), 1)), 1)
print(y_disp)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Accuracy Score
score = accuracy_score(y_test, y_pred)
print(score)


def logistic_visualization(x, y):
    """
    Visualizes the decision boundary for a logistic regression classifier using the global classifier and scaler.

    Parameters:
        x (ndarray): Feature set (scaled).
        y (ndarray): Labels.
    """
    # Inverse transform the scaled feature set for visualization
    x_set = sc.inverse_transform(x)
    y_set = y

    # Create a grid for the decision boundary
    X1, X2 = np.meshgrid(
        np.arange(start=x_set[:, 0].min() - 10, stop=x_set[:, 0].max() + 10, step=0.25),
        np.arange(start=x_set[:, 1].min() - 10000, stop=x_set[:, 1].max() + 10000, step=0.25)
    )

    # Predict and reshape for contour plotting
    plt.contourf(
        X1, X2,
        classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
        alpha=0.75, cmap=ListedColormap(('red', 'green'))
    )

    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    # Scatter plot for actual data points
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(
            x_set[y_set == j, 0], x_set[y_set == j, 1],
            c=ListedColormap(('red', 'green'))(i), label=j
        )

    plt.title('Logistic Regression Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


logistic_visualization(x_train, y_train)
logistic_visualization(x_test, y_test)

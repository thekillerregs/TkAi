import os

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


def resource_path(filename: str) -> str:
    """Return the absolute path to a file in the 'resources' folder."""
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Parent directory of the Python folder
    return os.path.join(base_dir, 'resources', filename)


# Importing data
dataset = pd.read_csv(resource_path('Churn_Modelling.csv'))

x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding Categorical Data
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Splitting dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Initializing ANN
ann = tf.keras.models.Sequential()
# Input
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# Hidden
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# Output
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Training
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(x_train, y_train, batch_size=32, epochs=100)

# Predicting!
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
y_pred = (ann.predict(x_test) > 0.5)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

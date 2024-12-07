import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np

def resource_path(filename: str) -> str:
    """Return the absolute path to a file in the 'resources' folder."""
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Parent directory of the Python folder
    return os.path.join(base_dir, 'resources', filename)


# Importing data, Preprocessing
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
training_set = train_datagen.flow_from_directory(resource_path('cnn/training_set'), target_size=(64, 64), batch_size=32,
                                                 class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory(resource_path('cnn/test_set'), target_size=(64, 64), batch_size=32,
                                            class_mode='binary')

cnn = tf.keras.models.Sequential()
# Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
# Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Second layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flattening
cnn.add(tf.keras.layers.Flatten())

# Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Output
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cnn.fit(x=training_set, validation_data=test_set, epochs=25)

# Single prediction
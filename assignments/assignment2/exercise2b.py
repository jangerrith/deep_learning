from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from tensorflow.keras import backend as K
import time
import numpy as np
import matplotlib.pyplot as plt

# This needs to be used prevent kernel crash for Mac OS
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Load mnist dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images.
x_train = (x_train / 255)
x_test = (x_test / 255)

# Determine image size
image_dim = x_train[0].shape
img_rows = image_dim[0]
img_cols = image_dim[1]

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape_ = (img_rows, img_cols, 1)

# Set seed to be used for random number generation
seed_ = 6570518

# Start time measurement
t0 = time.time()

# Build the model and its layers;
# note that input/ output dimensions (apart from the input layer)
# are inferred automatically
model = Sequential([
    Conv2D(32, kernel_size=(4, 4), strides=(2, 2), padding='same',
           activation='relu', kernel_initializer=keras.initializers.TruncatedNormal(
        seed=seed_), bias_initializer=keras.initializers.TruncatedNormal(seed=seed_),  input_shape=input_shape_),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
    Conv2D(16, kernel_size=(4, 4), strides=(1, 1), padding='same',
           activation='relu', kernel_initializer=keras.initializers.TruncatedNormal(
        seed=seed_), bias_initializer=keras.initializers.TruncatedNormal(seed=seed_),  input_shape=input_shape_),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
    Conv2D(9, kernel_size=(2, 2), strides=(1, 1), padding='same',
           activation='relu', kernel_initializer=keras.initializers.TruncatedNormal(
        seed=seed_), bias_initializer=keras.initializers.TruncatedNormal(seed=seed_),  input_shape=input_shape_),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
    Flatten(),  # flatten image to proceed with fully connected layer
    Dense(128, activation='sigmoid', kernel_initializer=keras.initializers.TruncatedNormal(
        seed=seed_), bias_initializer=keras.initializers.TruncatedNormal(seed=seed_)),
    Dense(10, activation='softmax'),
])

model.summary()

# Compile the model
# Note that categorical_crossentropy is the cross entropy function optimised for one-hot vectors that are used here
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(learning_rate=0.05),
              metrics=['accuracy'])

# Train the model.
# Note that to_categorical automatically transforms the labels into one-hot vectors
model.fit(x_train,
          to_categorical(y_train),
          batch_size=100,
          epochs=50,
          verbose=0)

# Calculate time in minutes the training took
t1 = time.time() - t0
print("Training model took " + str(round(t1 / 60)) + " minutes.")

# Evaluate current model on test data
final_loss = model.evaluate(
    x_test,
    to_categorical(y_test),
    verbose=0
)

# print('Validation loss on model: ' + str(val_loss))
print('Final test loss on model: ' + str(final_loss[0]))
print('Final test accuracy  on model: ' + str(final_loss[1]))

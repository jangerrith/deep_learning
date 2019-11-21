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

# Specify folds for cross-Validation
N_folds = 5

# Initialise k-fold cross-validation with N_folds folds, shuffling and our seed
kfold = KFold(n_splits=N_folds, shuffle=True, random_state=seed_)
# kfold = StratifiedKFold(n_splits=N_folds, shuffle=True, random_state=seed_)

t0 = time.time()

# Initialise vector to evaluate avg validation losses
val_losses = []

# Start cross-validation
for train_idx, val_idx in kfold.split(x_train):

    # Build the model and its layers;
    # note that input/ output dimensions (apart from the input layer)
    # are inferred automatically

    # Assign training and validation data
    x_train_CV, y_train_CV, = x_train[train_idx], y_train[train_idx]
    x_val_CV, y_val_CV, = x_train[val_idx], y_train[val_idx]

    # Initialise model
    model = Sequential([
        Conv2D(6, kernel_size=(5, 5), strides=(2, 2), padding='same',
               activation='relu', kernel_initializer=keras.initializers.TruncatedNormal(
                   seed=seed_), bias_initializer=keras.initializers.TruncatedNormal(seed=seed_),  input_shape=input_shape_),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
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
    model.fit(x_train_CV,
              to_categorical(y_train_CV),
              batch_size=100,
              epochs=50,
              verbose=0)

    # Evaluate current model on validation data
    loss_val = model.evaluate(
        x_val_CV,
        to_categorical(y_val_CV),
        verbose=0
    )

    # Compute different validation losses, one per split
    val_losses.append(loss_val[0])

    # exit cross-validation early
    break

# Calculate time in minutes the training took
t1 = time.time() - t0
print("Training model took " + str(round(t1 / 60)) + " minutes.")

# Evaluate current model on test data
final_loss = model.evaluate(
    x_test,
    to_categorical(y_test),
    verbose=0
)

# / N_folds // NOTE THAT WE EXIT THE CROSS-VALIDATION EARLY
avg_val_loss = sum(val_losses)
print('Average validation loss on model: ' + str(avg_val_loss))
print('Final test loss on model: ' + str(final_loss[0]))
print('Final test accuracy  on model: ' + str(final_loss[1]))

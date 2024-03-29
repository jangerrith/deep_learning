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

best_loss = 100
best_model = 10

# Initialise the models
model1 = Sequential([
    Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding='same',
           activation='relu', kernel_initializer=keras.initializers.TruncatedNormal(
               seed=seed_), bias_initializer=keras.initializers.TruncatedNormal(seed=seed_),  input_shape=input_shape_),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
    Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
           activation='relu', kernel_initializer=keras.initializers.TruncatedNormal(
        seed=seed_), bias_initializer=keras.initializers.TruncatedNormal(seed=seed_),  input_shape=input_shape_),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
    Flatten(),  # flatten image to proceed with fully connected layer
    Dense(128, activation='sigmoid', kernel_initializer=keras.initializers.TruncatedNormal(
        seed=seed_), bias_initializer=keras.initializers.TruncatedNormal(seed=seed_)),
    Dense(10, activation='softmax'),
])

model2 = Sequential([
    Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding='same',
           activation='relu', kernel_initializer=keras.initializers.TruncatedNormal(
               seed=seed_), bias_initializer=keras.initializers.TruncatedNormal(seed=seed_),  input_shape=input_shape_),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
    Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
           activation='relu', kernel_initializer=keras.initializers.TruncatedNormal(
        seed=seed_), bias_initializer=keras.initializers.TruncatedNormal(seed=seed_),  input_shape=input_shape_),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
    Flatten(),  # flatten image to proceed with fully connected layer
    Dense(256, activation='sigmoid', kernel_initializer=keras.initializers.TruncatedNormal(
        seed=seed_), bias_initializer=keras.initializers.TruncatedNormal(seed=seed_)),
    Dense(128, activation='sigmoid', kernel_initializer=keras.initializers.TruncatedNormal(
        seed=seed_), bias_initializer=keras.initializers.TruncatedNormal(seed=seed_)),
    Dense(10, activation='softmax'),
])

model3 = Sequential([
    Conv2D(16, kernel_size=(7, 7), strides=(1, 1), padding='same',
           activation='relu', kernel_initializer=keras.initializers.TruncatedNormal(
               seed=seed_), bias_initializer=keras.initializers.TruncatedNormal(seed=seed_),  input_shape=input_shape_),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
    Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same',
           activation='relu', kernel_initializer=keras.initializers.TruncatedNormal(
        seed=seed_), bias_initializer=keras.initializers.TruncatedNormal(seed=seed_),  input_shape=input_shape_),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
    Flatten(),  # flatten image to proceed with fully connected layer
    Dense(256, activation='sigmoid', kernel_initializer=keras.initializers.TruncatedNormal(
        seed=seed_), bias_initializer=keras.initializers.TruncatedNormal(seed=seed_)),
    Dense(128, activation='sigmoid', kernel_initializer=keras.initializers.TruncatedNormal(
        seed=seed_), bias_initializer=keras.initializers.TruncatedNormal(seed=seed_)),
    Dense(10, activation='softmax'),
])

# Compile the model
# Note that categorical_crossentropy is the cross entropy function optimised for one-hot vectors that are used here
model1.compile(loss='categorical_crossentropy',
               optimizer=keras.optimizers.SGD(learning_rate=0.05),
               metrics=['accuracy'])

model2.compile(loss='categorical_crossentropy',
               optimizer=keras.optimizers.SGD(learning_rate=0.05),
               metrics=['accuracy'])

model3.compile(loss='categorical_crossentropy',
               optimizer=keras.optimizers.SGD(learning_rate=0.05),
               metrics=['accuracy'])

# Loop over different models
for model_counter in range(1, 4):

    print('Training model ' + str(model_counter))
    t0 = time.time()

    # Note that we train the model only on one training/ validation data split
    val_loss = 0

    # Start cross-validation
    for train_idx, val_idx in kfold.split(x_train):

        # Build the model and its layers;
        # note that input/ output dimensions (apart from the input layer)
        # are inferred automatically

        # Assign training and validation data
        x_train_CV, y_train_CV, = x_train[train_idx], y_train[train_idx]
        x_val_CV, y_val_CV, = x_train[val_idx], y_train[val_idx]

        # Check which model to use
        if model_counter == 1:
            model = model1
        elif model_counter == 2:
            model = model2
        elif model_counter == 3:
            model = model3

        model.summary()

        # Train the model.
        # Note that to_categorical automatically transforms the labels into one-hot vectors
        history = model.fit(x_train_CV,
                            to_categorical(y_train_CV),
                            batch_size=100,
                            epochs=50,
                            verbose=0)

        # Summarize and plot history for loss
        plt.figure(1)
        plt.plot(history.history['loss'])
        plt.title('model loss over epochs')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(["Model 1", "Model 2", "Model 3"], loc='upper right')
        plt.savefig('./graphics/loss_over_epochs_exercise_2a.png',
                    bbox_inches='tight')

        # Evaluate current model on validation data
        loss_val = model.evaluate(
            x_val_CV,
            to_categorical(y_val_CV),
            verbose=0
        )

        # Compute different validation losses, one per split
        val_loss = loss_val[0]

        # Exit cross-validation early
        break

    # Calculate time in minutes the training took
    t1 = time.time() - t0
    print("Training model " + str(model_counter) +
          " took " + str(round(t1 / 60)) + " minutes and " + str(round(t1 % 60)) + " seconds.")

    # / N_folds // NOTE THAT WE EXIT THE CROSS-VALIDATION EARLY
    print('Average validation loss on model ' +
          str(model_counter) + ': ' + str(val_loss))

    # Select best model
    if val_loss < best_loss:
        best_loss = val_loss
        best_model = model_counter

print('The best performing model is model number ' + str(best_model) + '.\n ')

# Check which model to use
if best_model == 1:
    model = model1
elif best_model == 2:
    model = model2
elif best_model == 3:
    model = model3

# Evaluate current model on test data
final_loss = model.evaluate(
    x_test,
    to_categorical(y_test),
    verbose=0
)

print('Final test loss on model: ' + str(final_loss[0]))
print('Final test accuracy  on model:' + str(final_loss[1]))

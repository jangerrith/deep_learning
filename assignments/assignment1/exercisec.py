import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import time


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
pixels = image_dim[0] * image_dim[1]

# Flatten the images.
x_train = x_train.reshape((-1, pixels))
x_test = x_test.reshape((-1, pixels))

# Set seed to be used for random number generation
seed_ = 6570518

# Specify folds for cross-Validation
N_folds = 5

# Initialise k-fold cross-validation with N_folds folds, shuffling and our seed
kfold = KFold(n_splits=N_folds, shuffle=True, random_state=seed_)
# kfold = StratifiedKFold(n_splits=N_folds, shuffle=True, random_state=seed_)

best_model = 0
best_loss = 100


# Loop over different models
for model_counter in range(1, 6):

    print('Training model ' + str(model_counter))
    t0 = time.time()

    # Initialise vector to evaluate avg validation losses
    val_losses = []

    # Start cross-validation
    for train_idx, val_idx in kfold.split(x_train):

        # Build the model and its layers;
        # note that input/ output dimensions (apart from the input layer)
        # are inferred automatically
        model1 = Sequential([
            Dense(128, activation='sigmoid', input_shape=(pixels,), kernel_initializer=keras.initializers.TruncatedNormal(
                seed=seed_), bias_initializer=keras.initializers.TruncatedNormal(seed=seed_)),
            Dense(10, activation='softmax')
        ])

        model2 = Sequential([
            Dense(256, activation='sigmoid', input_shape=(pixels,), kernel_initializer=keras.initializers.TruncatedNormal(
                seed=seed_), bias_initializer=keras.initializers.TruncatedNormal(seed=seed_)),
            Dense(10, activation='softmax')
        ])

        model3 = Sequential([
            Dense(512, activation='sigmoid', input_shape=(pixels,), kernel_initializer=keras.initializers.TruncatedNormal(
                seed=seed_), bias_initializer=keras.initializers.TruncatedNormal(seed=seed_)),
            Dense(10, activation='softmax')
        ])

        model4 = Sequential([
            Dense(64, activation='sigmoid', input_shape=(pixels,), kernel_initializer=keras.initializers.TruncatedNormal(
                seed=seed_), bias_initializer=keras.initializers.TruncatedNormal(seed=seed_)),
            Dense(32, activation='sigmoid', kernel_initializer=keras.initializers.TruncatedNormal(
                seed=seed_), bias_initializer=keras.initializers.TruncatedNormal(seed=seed_)),
            Dense(10, activation='softmax')
        ])

        model5 = Sequential([
            Dense(256, activation='sigmoid', input_shape=(pixels,), kernel_initializer=keras.initializers.TruncatedNormal(
                seed=seed_), bias_initializer=keras.initializers.TruncatedNormal(seed=seed_)),
            Dense(128, activation='sigmoid', kernel_initializer=keras.initializers.TruncatedNormal(
                seed=seed_), bias_initializer=keras.initializers.TruncatedNormal(seed=seed_)),
            Dense(10, activation='softmax')
        ])

        # Check which model to use
        if model_counter == 1:
            model = model1
        elif model_counter == 2:
            model = model2
        elif model_counter == 3:
            model = model3
        elif model_counter == 4:
            model = model4
        elif model_counter == 5:
            model = model5

        # Assign training and validation data
        x_train_CV, y_train_CV, = x_train[train_idx], y_train[train_idx]
        x_val_CV, y_val_CV, = x_train[val_idx], y_train[val_idx]

        # Compile the model
        # Note that categorical_crossentropy is the cross entropy function optimised for one-hot vectors that are used here
        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=0.5),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        # Train the model.
        # Note that to_categorical automatically transforms the labels into one-hot vectors
        history = model.fit(
            x_train_CV,
            to_categorical(y_train_CV),
            epochs=50,
            batch_size=100,
            verbose=0
        )

        # Evaluate current model on validation data
        loss_val = model.evaluate(
            x_val_CV,
            to_categorical(y_val_CV),
            verbose=0
        )

        # Compute different validation losses, one per split
        val_losses.append(loss_val[0])

        print("Finished one split.")

    # Calculate time in minutes the training took
    t1 = time.time() - t0
    print("Training model " + str(model_counter) +
          " took " + str(round(t1 / 60)) + " minutes.")

    # Evaluate current model on test data
    final_loss = model.evaluate(
        x_test,
        to_categorical(y_test),
        verbose=0
    )

    avg_val_loss = sum(val_losses) / N_folds
    print('Average validation loss on model ' +
          str(model_counter) + ': ' + str(avg_val_loss))
    print('Final test loss on model ' +
          str(model_counter) + ': ' + str(final_loss[0]))
    print('Final test accuracy  on model ' +
          str(model_counter) + ': ' + str(final_loss[1]))

    # Select best model
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        best_model = model_counter

print('The best performing model is model number ' + str(best_model) + '.\n ')

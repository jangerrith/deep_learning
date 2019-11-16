import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np

# prevent kernel crash for Mac OS
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Load mnist dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

length_train = len(x_train)
length_test = len(x_test)

print("Amount of training data: " + str(length_train))
print("Amount of test data: " + str(length_test) + "\n")
print("\n")

# Normalize the images.
x_train = (x_train / 255)
x_test = (x_test / 255)

# Determine image size
image_dim = x_train[0].shape
pixels = image_dim[0] * image_dim[1]

# Flatten the images.
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

# Set initializer
seed_ = 6570518

# Specify folds for cross-Validation
N_folds = 5

# Initialise k-fold cross-validation with N_folds folds, shuffling and our seed
kfold = KFold(n_splits=N_folds, shuffle=True, random_state=seed_)

# Initialise vector to evaluate avg validation losses
val_losses = []

# Set counter to enumerate cross-validation splits
counter = 1

# Start cross-validation
for train_idx, val_idx in kfold.split(x_train):

    # Assign training and validation data
    x_train_CV, y_train_CV, = x_train[train_idx], y_train[train_idx]
    x_val_CV, y_val_CV, = x_train[val_idx], y_train[val_idx]

    print("Training with split " + str(counter))

    # Build the model
    model = Sequential([
        Dense(256, activation='sigmoid', input_shape=(pixels,), kernel_initializer=keras.initializers.TruncatedNormal(
            seed=seed_), bias_initializer=keras.initializers.TruncatedNormal(seed=seed_)),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    # Compile the model.
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.5),
        # use the categorical_crossentropy since we're using one-hot vectors for the labels
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    # Train the model.
    history = model.fit(
        x_train_CV,
        to_categorical(y_train_CV),  # convert the labels into one-hot vectors
        epochs=50,
        batch_size=100,
        verbose=0
    )

    # Summarize and plot history for loss
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.title('model loss over epochs')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['split 1', 'split 2', 'split 3',
                'split 4', 'split 5'], loc='upper right')
    plt.savefig('./graphs/loss_over_epochs_exercise_d.png',
                bbox_inches='tight')

    # Evaluate model on validation data
    loss_val = model.evaluate(
        x_val_CV,
        to_categorical(y_val_CV),  # convert the labels into one-hot vectors
        verbose=0
    )

    # Collect different validation losses, one per split
    val_losses.append(loss_val[0])

    # Increase split counter
    counter += 1

# Evaluate the model on (untouched) test data
final_loss = model.evaluate(
    x_test,
    to_categorical(y_test),  # convert the labels into one-hot vectors
    verbose=0
)

print('\n')
print('Average validation loss: ' + str(sum(val_losses) / N_folds))
print("Final test loss: " + str(final_loss[0]))
print("Final test accuracy: " + str(final_loss[1]))

# import tensorflow as tf
# disable tensorflow 2.0 behaviour
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import math
import sys

from sklearn.model_selection import KFold

# run 'conda install scikit-learn' first and then import cross
# validation functionality

mnist = tf.keras.datasets.mnist

#=================
# Input parameters
#=================
learningRate = 0.5
fc1_size = 128
batch_size = 100
N_epochs = 5
N_folds = 5
# number of splits for cross validation

#===========
# Load data
#===========

(x_train,y_train),(x_test,y_test) = mnist.load_data()
y_test_int = y_test
with tf.Session() as sess:
# with tf.compat.v1.Session() as sess:
    y_train = sess.run(tf.one_hot(y_train,10))
    y_test = sess.run(tf.one_hot(y_test,10))

x_train = x_train/255
x_test = x_test/255

imsize = x_train[0].shape
N_labels = 10
N_samples_train = len(x_train)

#=============
# Create model
#=============

X = tf.placeholder(tf.float32, shape = [None, imsize[0], imsize[1]])
y = tf.placeholder(tf.float32, shape = [None, N_labels])

#Hidden layer
with tf.variable_scope('fully_connected1') as scope:
    xdata = tf.reshape(X,[-1,imsize[0]*imsize[1]])
    W = tf.get_variable('weights',[imsize[0]*imsize[1],fc1_size],initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases',[fc1_size],initializer=tf.truncated_normal_initializer())
    fc1 = tf.nn.sigmoid(tf.matmul(xdata,W) + b)

#Output layer
with tf.variable_scope('output') as scope:
    W = tf.get_variable('weights',[fc1_size,N_labels],initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases',[N_labels],initializer=tf.truncated_normal_initializer())
    yhat = tf.squeeze(tf.matmul(fc1,W) + b) # squeeze gets rid of redundant dimensions

#Loss calculation
with tf.name_scope('loss'):
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y,logits=yhat)

#Optimization
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

pred = tf.argmax(tf.nn.softmax(yhat),1)

#=====================
# Training and testing
#=====================

with tf.Session() as sess:

    kf = KFold(n_splits=N_folds, shuffle=True, random_state=9812938)
    # makes N_folds lists of indices

    val_losses = []
    # will be a vector of length N_folds, collecting
    # different validation loss values

    for train_idx, val_idx in kf.split(x_train):
        # train_idx, val_idx are lists of identifiers returned by
        # kf.split(x_train)

        x_train_CV, y_train_CV, = x_train[train_idx], y_train[train_idx]
        x_val_CV, y_val_CV, = x_train[val_idx], y_train[val_idx]
        # this just pulls out training samples + labels
        # and validation samples + labels

        N_samples_train_CV = x_train_CV.shape[0]
        N_samples_val_CV = x_val_CV.shape[0]
        # compute numbers of samples in training and validation data

        # all that follows is just as usual, with predictions on validation data
        sess.run(tf.global_variables_initializer())
        items = np.arange(N_samples_train_CV) # N_samples_train earlier
        for epoch in range(N_epochs):
            np.random.shuffle(items)
            N_batches = math.ceil(N_samples_train_CV/batch_size)
            idx_start = 0
            for bat in range(N_batches-1):
                x_train_batch = x_train_CV[items[idx_start:idx_start+batch_size]]
                y_train_batch = y_train_CV[items[idx_start:idx_start+batch_size]]
                _, loss_train = sess.run([optimizer, loss], feed_dict={X:x_train_batch,y:y_train_batch})
                idx_start += batch_size
            x_train_batch = x_train_CV[items[idx_start:]]
            y_train_batch = y_train_CV[items[idx_start:]]
            _, loss_train = sess.run([optimizer, loss], feed_dict={X:x_train_batch,y:y_train_batch})

            print('Training loss: '+str(loss_train))

        pred_,loss_val = sess.run([pred,loss], feed_dict={X:x_val_CV,y:y_val_CV})
        print('Validation loss: '+str(loss_val))
        val_losses.append(loss_val)
        # compute different validation losses, one per split

    print('Average validation loss: '+str(sum(val_losses)/N_folds))
    # one can now decide when to stop training or choosing
    # architectures by looking at validation losses (or accuracies)

    pred_,loss_test = sess.run([pred,loss], feed_dict={X:x_test,y:y_test})
    print('Test loss: '+str(loss_test))

    compare = [pred_[i] == y_test_int[i] for i in range(len(y_test_int))]
    acc = sum(compare)/len(y_test_int)
    print('Test accuracy: '+str(acc))

import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist

import sys

seed_=1

#=================
# Input parameters
#=================
learningRate = 0.5
fc1_size = 1024

#===========
# Load data
#===========

(x_train,y_train),(x_test,y_test) = mnist.load_data()
with tf.Session() as sess:
    y_train = sess.run(tf.one_hot(y_train,10))
    y_test = sess.run(tf.one_hot(y_test,10))
    # this turns y_train,y_test from numpy arrays into one_hot vectors
    
x_train = x_train/255
x_test = x_test/255
# puts x values (grayscale) into [0,1]

imsize = x_train[0].shape
N_labels = 10

#=============
# Create model
#=============

X = tf.placeholder(tf.float32, shape = [None, imsize[0], imsize[1]])
y = tf.placeholder(tf.float32, shape = [None, N_labels])

#Hidden layer
with tf.variable_scope('fully_connected1') as scope:
    xdata = tf.reshape(X,[-1,imsize[0]*imsize[1]])
    W = tf.get_variable('weights',[imsize[0]*imsize[1],fc1_size],initializer=tf.truncated_normal_initializer(seed=seed_))
    b = tf.get_variable('biases',[fc1_size],initializer=tf.truncated_normal_initializer(seed=seed_))
    fc1 = tf.nn.sigmoid(tf.matmul(xdata,W) + b)
    # changes from sigmoid to softmax to speed up training: reversed to sigmoid

#Output layer
with tf.variable_scope('output') as scope:
    W = tf.get_variable('weights',[fc1_size,N_labels],initializer=tf.truncated_normal_initializer(seed=seed_))
    b = tf.get_variable('biases',[N_labels],initializer=tf.truncated_normal_initializer(seed=seed_))
    yhat = tf.squeeze(tf.matmul(fc1,W) + b)

#Loss calculation
with tf.name_scope('loss'):
    loss = tf.losses.sigmoid_cross_entropy(y,yhat)
    # adapt loss as well to fit sigmoid as activation function

#Optimization
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

#=====================
# Training and testing
#=====================

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    _, loss_train = sess.run([optimizer, loss], feed_dict={X:x_train,y:y_train})
    print('Training loss: '+str(loss_train))

    yhat_,loss_test = sess.run([yhat,loss], feed_dict={X:x_test,y:y_test})
    print('Test loss: '+str(loss_test))

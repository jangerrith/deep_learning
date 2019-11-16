import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist

import sys

#=================
# Input parameters
#=================
learningRate = 0.5
fc1_size = 1024

#===========
# Load data
#===========

(x_train,y_train),(x_test,y_test) = mnist.load_data()
# that's numpy arrays in return

imsize = x_train[0].shape
# shape of first training image, which is a tuple of integers (28,28)

N_labels = 1
# number of labels

#=============
# Create model
#=============

X = tf.placeholder(tf.float32, shape = [None, imsize[0], imsize[1]])
y = tf.placeholder(tf.float32, shape = [None])

#Hidden layer
with tf.variable_scope('fully_connected1') as scope:
    
    # create scope, referring to the first fully connected layer,
    # in which W, b, fc1 all live in, internally think of
    # names fully_connected1/W fully_connected1/b and so on
    
    xdata = tf.reshape(X,[-1,imsize[0]*imsize[1]])
    # same like xdata = tf.placeholder(tf.float32, shape = [None,
    # imsize[0]*imsize[1])
    
    W = tf.get_variable('weights',[imsize[0]*imsize[1],fc1_size],initializer=tf.truncated_normal_initializer())
    
    # W is for the weights from input layer of size
    # imsize[0]*imsize[1] to first hidden layer of size fc1_size,

    # weights are initialized as told 'weights' is a name for use in
    # TensorBoard
    
    b = tf.get_variable('biases',[fc1_size],initializer=tf.truncated_normal_initializer())
    fc1 = tf.nn.sigmoid(tf.matmul(xdata,W) + b)
    
    # generates values for first hidden layer, applying sigmoid to
    # (W*xdata+b)

#Output layer
with tf.variable_scope('output') as scope:
    W = tf.get_variable('weights',[fc1_size,N_labels],initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases',[N_labels],initializer=tf.truncated_normal_initializer())
    yhat = tf.squeeze(tf.matmul(fc1,W) + b)
    
    # represents predicted output, without activation function yet
    # squeeze gets rid of redundant dimensions

    # the different scopes allow to reuse variable names (W,b) and
    # combine with variables from other scopes (fc1, the output of the
    # earlier layer)

#Loss calculation
with tf.name_scope('loss'):
    loss = tf.losses.mean_squared_error(y,yhat)
    # loss function is mean squared error
    # one can possibly leave away name_scope

#Optimization
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)
# setting up the optimization procedure

#=====================
# Training and testing
#=====================

with tf.Session() as sess: # above you set up the framework, and here you use, you do the stuff
    
    sess.run(tf.global_variables_initializer())
    # initializes all variables with initializers as specified above
    
    _, loss_train = sess.run([optimizer, loss], feed_dict={X:x_train,y:y_train})
    # _ is for passing on values to nowhere, only loss_train is of real interest
    # feed_dict: x_train, y_train should fit the shapes of the placeholders from above
    
    print('Training loss: '+str(loss_train))

    yhat_,loss_test = sess.run([yhat,loss], feed_dict={X:x_test,y:y_test})
    # inserting yhat in sess.run creates the ability to look at the predicted labels
    # for the moment, we don't need it, so passed on to yhat_
    
    print('Test loss: '+str(loss_test))

    # NOTE: this is only *one* epoch overall, so results are rather bad

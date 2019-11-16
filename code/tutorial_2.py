import tensorflow as tf
import numpy as np
import math
import sys
mnist = tf.keras.datasets.mnist

#=================
# Input parameters
#=================
learningRate = 0.5
fc1_size = 1024
batch_size = 100 # new: work with batches

#===========
# Load data
#===========

(x_train,y_train),(x_test,y_test) = mnist.load_data()
with tf.Session() as sess:
    y_train = sess.run(tf.one_hot(y_train,10))
    y_test = sess.run(tf.one_hot(y_test,10))

x_train = x_train/255
x_test = x_test/255

imsize = x_train[0].shape
N_labels = 10
N_samples_train = len(x_train)
# overall number of training samples

#=============
# Create model
#=============

X = tf.placeholder(tf.float32, shape = [None, imsize[0], imsize[1]])
y = tf.placeholder(tf.float32, shape = [None,N_labels])

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
    yhat = tf.squeeze(tf.matmul(fc1,W) + b)
    # note that N_labels has changed from 1 to 10, to account for the
    # one-hot-vectors

#Loss calculation
with tf.name_scope('loss'):
    loss = tf.losses.sigmoid_cross_entropy(y,yhat)

#Optimization
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

#=====================
# Training and testing
#=====================

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    items = np.arange(N_samples_train)
    # items == array([0,1,...,N_samples_train-1])
    
    np.random.shuffle(items)
    # now items == array([ ... shuffled version of items ...])
    
    N_batches = math.ceil(N_samples_train/batch_size)
    # N_batches is the number of batches
    
    idx_start = 0
    for bat in range(N_batches-1):
        # now train with every batch, accordingly load training data into X and y
        
        x_train_batch = x_train[items[idx_start:idx_start+batch_size]]
        y_train_batch = y_train[items[idx_start:idx_start+batch_size]]
        
        _, loss_train = sess.run([optimizer, loss], feed_dict={X:x_train_batch,y:y_train_batch})
        # now load the training batch into X and y, and train
        
        idx_start += batch_size
        # iterate, and train on every batch
        
    x_train_batch = x_train[items[idx_start:]]
    y_train_batch = y_train[items[idx_start:]]
    # last batch, which may not be of size 100
    
    _, loss_train = sess.run([optimizer, loss], feed_dict={X:x_train_batch,y:y_train_batch})
    # train also on last batch
    
    print('Training loss: '+str(loss_train))

    yhat_,loss_test = sess.run([yhat,loss], feed_dict={X:x_test,y:y_test})
    print('Test loss: '+str(loss_test))

    # this is one epoch, but with the batches results should be a bit improved

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import math
import sys
mnist = tf.keras.datasets.mnist

#=================
# Input parameters
#=================
learningRate = 0.5
fc1_size = 256
batch_size = 100
N_epochs = 10

#===========
# Load data
#===========

(x_train,y_train),(x_test,y_test) = mnist.load_data()
y_test_int = y_test
# keeps original test data for later evaluation

with tf.Session() as sess:
    y_train = sess.run(tf.one_hot(y_train,10))
    y_test = sess.run(tf.one_hot(y_test,10))

x_train = x_train/255
x_test = x_test/255

imsize = x_train[0].shape
N_labels = 10
N_samples_train = len(x_train)

seed_ = 6570518

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

#Output layer
with tf.variable_scope('output') as scope:
    W = tf.get_variable('weights',[fc1_size,N_labels],initializer=tf.truncated_normal_initializer(seed=seed_))
    b = tf.get_variable('biases',[N_labels],initializer=tf.truncated_normal_initializer(seed=seed_))
    yhat = tf.squeeze(tf.matmul(fc1,W) + b)

#Loss calculation
with tf.name_scope('loss'):
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y,logits=yhat)
    # now changes from sigmoid+cross_entropy to softmax+cross-entropy
    # where arguments 'onehot_labels' and 'logits' need to be passed

#Optimization
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

pred = tf.argmax(tf.nn.softmax(yhat),1)
# that's for the test data, where no training is needed, 1 specifies
# that argmax is to be taken across columns, which refer to different
# softmax values per sample (=row)

#=====================
# Training and testing
#=====================

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    items = np.arange(N_samples_train)
    for epoch in range(N_epochs):
        np.random.shuffle(items)
        N_batches = math.ceil(N_samples_train/batch_size)
        idx_start = 0
        for bat in range(N_batches-1):
            x_train_batch = x_train[items[idx_start:idx_start+batch_size]]
            y_train_batch = y_train[items[idx_start:idx_start+batch_size]]
            _, loss_train = sess.run([optimizer, loss], feed_dict={X:x_train_batch,y:y_train_batch})
            idx_start += batch_size
        x_train_batch = x_train[items[idx_start:]]
        y_train_batch = y_train[items[idx_start:]]
        _, loss_train = sess.run([optimizer, loss], feed_dict={X:x_train_batch,y:y_train_batch})

        print('Training loss: '+str(loss_train))

    pred_,loss_test = sess.run([pred,loss], feed_dict={X:x_test,y:y_test})
    print('Test loss: '+str(loss_test))
    # Now predictions for test data are computed

    compare = [pred_[i] == y_test_int[i] for i in range(len(y_test_int))]
    # compare is a vector of length of the number of test data points,
    # where 1 as entry reflects that the prediction was correct, and 0
    # reflects that the prediction was wrong

    acc = sum(compare)/len(y_test_int)
    # dividing the number of correct predictions, which is the number
    # of 1's in compare by the number of test data points
    # results in the accuracy of the predictions

    print('Test accuracy: '+str(acc))
    # compute accuracy: could do this before so as to see accuracy for
    # only one epoch, one batch; one epoch, several batches

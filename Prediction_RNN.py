"""
Prediction a Time-Series with Recurrent Neural Network
*************************************************************
**Author**: `Amir Shirian <https://github.com/Amir_Shirian>`_
In this project we will be teaching a neural network to follow a time sequence.

This is made possible by the simple but powerful idea of the `LSTM
unit <http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf>`__, in which many
LSTM units work together to find a model to predict a sequence.
We can look at Prediction problem as regression. There are two approaches,
first, you can learn your network with one-by-one i-th data as input and 
(i+1)-th data as label. Second, you can consider a window of data ( (i-N)-th : (i-1)-th )
to predict i-th data. In this project we implement this two approaches.
**Recommended Reading:**
I assume you have at least installed tensorflow v.1.0.1, know Python, and
understand Tensors.
**Requirements**
To run: 
$ python Prediction_RNN.py --model=RNN --window_size=1 --train_size=0.70
With help of `Aymeric Damien`
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import batch_counting as batch
import matplotlib.pyplot as plt
from pandas import read_csv


flags = tf.flags

flags.DEFINE_string("model", "MultiRNN",
     "A type of model. Possible options are: RNN, biRNN, MultiRNN.")
flags.DEFINE_integer("window_size" , 4, "Window size" )
flags.DEFINE_float("train_size", 0.67, "Train Percentage %" )

FLAGS = flags.FLAGS


# Creating data

def f(x):
    return np.sin(x)
x=np.linspace(0,100,1000)
dataset=f(x)

# load the dataset
# dataframe = read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
# dataset = dataframe.values
# dataset = dataset.astype('float32')

plt.figure(0)
plt.plot(dataset)
plt.xlabel('samples')
plt.ylabel('Data')

# split data into train and test
Train_size = int(len(dataset) * FLAGS.train_size)
Test_size = len(dataset) - Train_size
train , test = dataset[0:Train_size], dataset[Train_size:len(dataset)]



# convert an array of values(time series) into a dataset matrix
def creat_dataset(dataset, window=1):
    dataX, dataY = [] , []
    for i in range(len(dataset) - window - 1):
        a = dataset[i:(i+window)]
        dataX.append(a)
        dataY.append(dataset[i + window])
    return np.array(dataX), np.array(dataY)

# assign length of window
window = FLAGS.window_size


# when we use window mode the size of training and test data will be changed
Train_size = Train_size - window - 1
Test_size = Test_size - window - 1

trainX , trainY = creat_dataset(train, window)
testX , testY = creat_dataset(test, window)

# Parameters
learning_rate = 0.001
training_iters = 1000000
batch_size = 5
display_step = 1000

# Network Parameters
n_input = window # size of data input 
n_steps = 1 # number of time series
n_hidden = 10 # hidden layer num of features
n = 1 # number of sample that we want predict
n_layers = 4

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n])


# Define weights
if FLAGS.model=='RNN':
    # init_state = tf.placeholder("float", [None, n_hidden])
    weights = {
        # Hidden layer weights => 2*n_hidden because of forward + backward cells
        'out': tf.Variable(tf.random_normal([n_hidden, n]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n]))
    }
if FLAGS.model=='biRNN':
    # init_state = tf.placeholder("float", [2, None, n_hidden])
    weights = {
        # Hidden layer weights => 2*n_hidden because of forward + backward cells
        'out': tf.Variable(tf.random_normal([2*n_hidden, n]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n]))
    }
if FLAGS.model=='MultiRNN':
    # init_state = tf.placeholder("float", [n_layers, 2, None, n_hidden])
    weights = {
        # Hidden layer weights => 2*n_hidden because of forward + backward cells
        'out': tf.Variable(tf.random_normal([n_hidden, n]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n]))
    }

# Define models
def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out'], states

def BiRNN(x, weights, biases):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    try:
        outputs, state_fw, state_bw = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out'], [ state_fw, state_bw ]

def MultiRNN(x, weights, biases, n_layers=1):
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Untack to get a list of 'n_layers' tensors of shape (2, batch_size, n_hidden)
    # state_per_layer_list = tf.unstack(init_state, axis=0)
    # rnn_tuple_state = tuple([rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
        # for idx in range(n_layers)])

    # Get multi lstm cells
    def build_lstm_cell():
        lstm_cell = rnn.LSTMCell(n_hidden)
        return lstm_cell
    cell = rnn.MultiRNNCell([build_lstm_cell() for _ in range(n_layers)])

    # Get cell output
    outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out'], states

    



# Select model
if FLAGS.model=='RNN':
    # _init_state = np.zeros((batch_size, n_hidden))
    pred, states = RNN(x, weights, biases)
if FLAGS.model=='biRNN':
    # _init_state = np.zeros((2, batch_size, n_hidden))
    pred, states = BiRNN(x, weights, biases)
if FLAGS.model=='MultiRNN':
    # _init_state = np.zeros((n_layers, 2, batch_size, n_hidden))
    pred, states = MultiRNN(x, weights, biases, n_layers)

# Define loss and optimizer
cost = tf.losses.mean_squared_error(labels=y, predictions=pred)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.losses.mean_squared_error(labels=y, predictions=pred)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# creat batch indices
sampler = batch.batchcounter( batch_size , Train_size)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        inds = sampler.next_inds()
        batch_x, batch_y = trainX[inds], trainY[inds]
        # Reshape data
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        batch_y = batch_y.reshape([batch_size, n])
        # Run optimization op (backprop)
        _states, _ = sess.run([states, optimizer], feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch loss
            loss = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate test loss
            # Reshape data
            testX = testX.reshape((testX.shape[0], n_steps, n_input))
            testY = testY.reshape([testY.shape[0], n])
            Predi = sess.run(pred, feed_dict={x: testX, y: testY})
            acc = sess.run(cost, feed_dict={x: testX, y: testY})
            print("Iter " + str(step*batch_size) + ", Test Loss= " + \
                  "{:.6f}".format(acc) + ", Training Loss= " + \
                  "{:.5f}".format(loss))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 100 sample of time series
    Predictions = sess.run(pred, feed_dict={x: testX, y: testY})
    MSE = (testY- Predictions)**2
    sigma = np.sqrt(MSE)
    plt.figure(1)
    plt.plot(test, 'b-', label=u'Observations')
    plt.plot( Predictions,'r:',label=u'Predictions')
    plt.fill(np.concatenate([range(Predictions.shape[0]), range(Predictions.shape[0])[::-1] ]),
         np.concatenate( [ Predictions - 1.96 *sigma ,( Predictions + 1.96*sigma )[::-1] ] ),
          alpha=0.5, fc='b', ec='None', label='95% confidence interval')
    # 2.59=99%  1.96=95%
    plt.legend(loc='upper left')
    plt.show()

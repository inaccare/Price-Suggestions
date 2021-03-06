# Import data
import pandas as pd
import re
import sys
import time
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import json
import os
# import tensorflow_utils
#from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

"""
This model's architecture is as follows: product descriptions' average w2v encodings go into
a 2-layer FC network. Output is a softmax vector of length 12, corresponding to 12 distinct
price buckets. Pre-trained glove-based w2v encodings used.
"""


def main():
    # Usage is as follows: python model.py <train_enc>.csv <dev_enc>.csv(optional)
    # Test variables are dummy variables for now
    X_test = []
    Y_test = []
    trainCSV = None
    devCSV = None
    if (len(sys.argv) >=2):
        trainCSV = sys.argv[1]
    if (len(sys.argv) >= 3):
        devCSV = sys.argv[2]
    trainDF = pd.read_csv(trainCSV, header = 0)
#    trainDF = trainDF.truncate(before = 0, after = 50000)
    trainDFBuckets = trainDF[['price-bucket']]
    X_train, Y_train = getProductEncodingsAndPrices(trainDF, trainDFBuckets)
    devDF = pd.read_csv(devCSV, header = 0)
    devDFBuckets = devDF[['price-bucket']]
    X_dev, Y_dev = getProductEncodingsAndPrices(devDF, devDFBuckets)
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_dev shape: " + str(X_dev.shape))
    print ("Y_dev shape: " + str(Y_dev.shape))
    parameters = model(X_train, Y_train, X_dev, Y_dev, num_epochs = 300)

def readInArray(word2vecList):
    """
    Reads in the word2vec vector for a given item and returns it in the proper format
    """
    word2vecList = str(word2vecList)
    vecList = []
    splitArr = word2vecList.split()
    for num in splitArr:
        if num[0] == '[':
            num = num[1:]
        elif num[-1] == ']':
            num = num[:-1]
        if len(num) > 1:
            vecList.append((float)(num))
    if sum(vecList) == 0:
        return np.zeros(len(vecList))
    return vecList

def OneHot(bucket, numBuckets):
    arr = np.zeros(numBuckets)
    arr[bucket] = 1
    return arr

# Function looks at the dataframe and returns matrix of description encodings for all samples such that X.shape = (n_x, m)
def getProductEncodingsAndPrices(dfEncodings, dfBuckets):
    X = []
    Y = []
    numBuckets = 12
    for i in range(0, len(dfEncodings['word2vec'])):
        X.append(readInArray(dfEncodings['word2vec'][i]))
        Y.append(OneHot(int(dfBuckets['price-bucket'][i]), numBuckets))
    Y= np.array(Y).T
    X = np.array(X).T
    return X, Y

def random_mini_batches(X, Y, mini_batch_size = 128, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[1]                 # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def model(X_train, Y_train, X_dev, Y_dev, learning_rate = 0.0001,
          num_epochs = 300, minibatch_size = 128, print_cost = True):
    """
    Implements a two-layer tensorflow neural network: LINEAR->RELU->LINEAR->SOFTMAX.
    Arguments:
    X_train -- training set, of shape (input size = 100, number of training examples = ~1.4M)
    Y_train -- test set, of shape (output size = 12, number of training examples = ~1.4M)
    X_test -- training set, of shape (input size = 100, number of training examples = 14825)
    Y_test -- test set, of shape (output size = 12, number of test examples = 14825)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 10 epochs
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) =  100, X_train.shape[1]                 # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)
    # Initialize parameters
    parameters = initialize_parameters(n_x, m, n_y)
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z4 = forward_propagation(X, parameters)
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z4, Y)
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    # Initialize all the variables
    init = tf.global_variables_initializer()
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)
        # Do the training loop
        for epoch in range(num_epochs):
            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict = {X:minibatch_X, Y:minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                #if (epoch != 0):
                #   os.remove('cost_nn_w2v_expand'+str(epoch-5) +'.json')
                #fileoutCost = open('cost_nn_w2v_expand' + str(epoch) +'.json', 'w')
                #json.dump(costs, fileoutCost)

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z4), tf.argmax(Y))
        # Calculate accuracy on the train set
        accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float"))

        Accuracy = (accuracy.eval({X: X_train, Y: Y_train}))/m
        print ('Train accuracy is ' + str(Accuracy))
        Accuracy = (accuracy.eval({X: X_dev, Y: Y_dev}))/(X_dev.shape[1])
        print ('Dev accuracy is ' + str(Accuracy))
        for val in parameters:
            parameters[val] = parameters[val].tolist()
        fileout = open('parameters_nn_w2v_expand.json', 'w')
        json.dump(parameters, fileout)
        fileoutCost = open('cost_nn_w2v_expand.json', 'w')
        json.dump(costs, fileoutCost)
        # plot the cost
        # plt.plot(np.squeeze(costs))
        # plt.ylabel('word2vec model cost')
        # plt.xlabel('iterations (per tens)')
        # plt.title("Learning rate = " + str(learning_rate))
        # plt.show()
        return parameters

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    ### START CODE HERE ### (approx. 2 lines)
    X = tf.placeholder('float32', [n_x, None], name = "X")
    Y = tf.placeholder('float32', [n_y, None], name = "Y")
    ### END CODE HERE ###

    return X, Y

def initialize_parameters(n_x, m, n_y):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    tf.set_random_seed(1)                   # so that your "random" numbers match ours

    ### START CODE HERE ### (approx. 6 lines of code)
    W1 = tf.get_variable("W1", [50,n_x], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [50, 1], initializer= tf.zeros_initializer())
    W2 = tf.get_variable("W2", [25, 50], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [25, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [n_y, 25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [n_y, 1], initializer = tf.zeros_initializer())
    #W4 = tf.get_variable("W4", [n_y, 15], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    #b4 = tf.get_variable("b4", [n_y, 1], initializer = tf.zeros_initializer())

    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}#,
                  # "W4": W4,
                  # "b4": b4}

    return parameters

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters
    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    # W4 = parameters['W4']
    # b4 = parameters['b4']

    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                                              # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                                              # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                                              # Z3 = np.dot(W3,Z2) + b3
    # A3 = tf.nn.relu(Z3)
    # Z4 = tf.add(tf.matmul(W4, A3), b4)
    ### END CODE HERE ###

    return Z3


def compute_cost(Z4, Y):
    """
    Computes the cost
    Arguments:
    Z2 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z2
    Returns:
    cost - Tensor of the cost function
    """

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z4)
    labels = tf.transpose(Y)

    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    ### END CODE HERE ###

    return cost

if __name__ == "__main__":
    main()

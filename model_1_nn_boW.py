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


def main():
    # Usage is as follows: python model.py <train_enc>.csv <dev_enc>.csv

    X_test = []
    Y_test = []
    devCSV = None
    trainCSV = None

    if (len(sys.argv) >=2):
        trainCSV = sys.argv[1]
    if (len(sys.argv) >= 3):
        devCSV = sys.argv[2]
    trainDF = pd.read_csv(trainCSV, header = 0)
    X_train, Y_train = getProductEncodingsAndPrices(trainDF)
    devDF = pd.read_csv(devCSV, header = 0)
    X_dev, Y_dev = getProductEncodingsAndPrices(devDF)
    print ("X_train shape: " + str(len(X_train)))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_dev shape: " + str(X_dev.shape))
    print ("Y_dev shape: " + str(Y_dev.shape))
    parameters = model(X_train, Y_train, X_dev, Y_dev, num_epochs = 5)

def random_mini_batches(X, Y, mini_batch_size = 128, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector of shape (numbuckets, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- so that everyone in our group will get same minibatches permutations

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = len(X)                 # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def expandMinibatchArrays(minibatch_X):
    """
    Expands the minibatch arrays from indices to binary vectors

    Arguments:
    minibatch_X -- training examples of the current minibatch
    Returns:
    mini_batch_vectors -- training expamples as binary vectors
    """
    X = []
    vocabLength = 517430  #Look at vocab-length for this value. Everytime bag_of_words is run, this file is written to.
    for sample in minibatch_X:
        X.append(expandArray(sample, vocabLength))
    return np.array(X).T


def getProductEncodingsAndPrices(df):
    """
    Function looks at the dataframe and returns matrix of description
    condensed encodings for all samples (each sample has variable length vectors
        of indices corresponding to vocab indices of words that appear in each
        description)

    Arguments:
    df -- dataframe of data
    Returns:
    X -- product descriptions
    Y -- price of item placed in pre-determined buckets
    """
    X = []
    Y = []
    numBuckets = 12
    for i in range(0, len(df['encodings'])):
        X.append(df['encodings'][i])
        Y.append(OneHot(int(df['price-bucket'][i]), numBuckets))
    Y= np.array(Y).T
    X = np.array(X).T
    return X, Y

def OneHot(bucket, numBuckets):
    """
    Creates onhot vector for our Y output

    Arguments:
    bucket -- index of correct bucket for example in Y
    numBuckets -- number of buckets used to split of the prices of objects
    Returns:
    arr -- onehot array
    """
    arr = np.zeros(numBuckets)
    arr[bucket] = 1
    return arr

def expandArray(List, vocabLength):
    """
    Turns encodings of variable size (related to sentence length) to encodings of
    vector length corresponding to length of vocab

    Arguments:
    List -- list of encodings as indices
    vocabLength -- the length of our vocabulary (corpus)
    Returns:
    arr -- expanded input vectors to be used in model
    """
    arr = np.zeros(vocabLength)
    # Have to do it like this because read_csv defaults the List to really be a string that visibly looks like a list (eg. "[0 1 3]")
    for i in List.split(' '):
        if i.isdigit():
            arr[(int)(i)] = 1
    return arr

def model(X_train, Y_train, X_dev, Y_dev, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 128, print_cost = True):
    """
    Implements a two-layer tensorflow neural network: LINEAR->RELU->LINEAR->SOFTMAX.
    Arguments:
    X_train -- training set, of shape (input size = vocab_length, number of training examples = 1452885)
    Y_train -- test set, of shape (output size = 12, number of training examples = 1452885)
    X_test -- training set, of shape (input size = vocab_length, number of test examples)
    Y_test -- test set, of shape (output size = 12, number of test examples)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) =  517430, len(X_train)                        # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters(n_x, m, n_y)

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z2 = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z2, Y)

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

                # The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                # Turns index array of variable length to one-hot array of length vocab-length
                minibatch_X = expandMinibatchArrays(minibatch_X)
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict = {X:minibatch_X, Y:minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)


        # Save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z2), tf.argmax(Y))

        # Calculate accuracy on the train set
        accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float"))
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
        overallRight = 0
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            minibatch_X = expandMinibatchArrays(minibatch_X)
            overallRight = overallRight + accuracy.eval({X: minibatch_X, Y: minibatch_Y})
        print ('Train accuracy = ', str(overallRight/m))
        # Calculate accuracy on the dev set
        minibatches = random_mini_batches(X_dev, Y_dev, minibatch_size, seed)
        overallRight = 0
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            minibatch_X = expandMinibatchArrays(minibatch_X)
            overallRight = overallRight + accuracy.eval({X: minibatch_X, Y: minibatch_Y})
        print ('Dev accuracy = ', str(overallRight/len(X_dev)))
        # Writing out trained parameters onto an uotput file called parameters.json
        for val in parameters:
            parameters[val] = parameters[val].tolist()
        fileout = open('parameters.json', 'w')
        json.dump(parameters, fileout)
        # plot the cost
        # plt.plot(np.squeeze(costs))
        # plt.ylabel('cost')
        # plt.xlabel('iterations (per tens)')
        # plt.title("Learning rate =" + str(learning_rate))
        # plt.show()
        return parameters

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    Arguments:
    n_x -- scalar, size of vocab
    n_y -- scalar, number of classes
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

    - Use None because it let's us be flexible on the number of examples you will for the placeholders
    """
    X = tf.placeholder('float32', [n_x, None], name = "X")
    Y = tf.placeholder('float32', [n_y, None], name = "Y")

    return X, Y

def initialize_parameters(n_x, m, n_y):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, n_x]
                        b1 : [25, 1]
                        W2 : [n_y, 25]
                        b2 : [n_y, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2
    """

    tf.set_random_seed(1)                   # so that your "random" numbers match ours

    W1 = tf.get_variable("W1", [25,n_x], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [25, 1], initializer= tf.zeros_initializer())
    W2 = tf.get_variable("W2", [n_y, 25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [n_y, 1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> SOFTMAX
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2"
                  the shapes are given in initialize_parameters
    Returns:
    Z2 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = tf.add(tf.matmul(W1, X), b1)                                              # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                                              # Z2 = np.dot(W2, a1) + b2

    return Z2


def compute_cost(Z2, Y):
    """
    Computes the cost
    Arguments:
    Z2 -- output of forward propagation (output of the last LINEAR unit)
    Y -- "true" labels vector placeholder, same shape as Z2
    Returns:
    cost - Tensor of the cost function
    """

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z2)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    return cost

if __name__ == "__main__":
    main()

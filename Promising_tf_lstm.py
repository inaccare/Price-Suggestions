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
import nltk


w2v = None
gloveFile = sys.argv[3]
with open(gloveFile, "rb") as lines:
    w2v = {line.split()[0]: np.array(line.split()[1:])
            for line in lines}
wordToIndex = dict()
indexToEmb = dict()
count = 1
for w in w2v:
    wordToIndex[w] = count
    indexToEmb[count] = w2v[w]
    count = count + 1

def main():
    # Usage is as follows: python model.py <train_enc>.csv <dev_enc>.csv <glove file>

    X_test = []
    Y_test = []
    testCSV = None
    trainCSV = None
    if (len(sys.argv) >=2):
        trainCSV = sys.argv[1]
    if (len(sys.argv) == 4):
        devCSV = sys.argv[2]
    trainDF = pd.read_csv(trainCSV, header = 0)
    # For each entry in X_train, we have an array of length T_x with each entry
    # corresponding to an index into the word's w2v embedding
    X_train, Y_train = getProductIndicesAndPrices(trainDF)
    devDF = pd.read_csv(devCSV, header = 0)
    X_dev, Y_dev = getProductIndicesAndPrices(devDF)
    print ("X_train shape: " + str(len(X_train)))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_dev shape: " + str(X_dev.shape))
    print ("Y_dev shape: " + str(Y_dev.shape))

    model(X_train, Y_train, X_dev, Y_dev)

def getProductIndicesAndPrices(df):
    X = []
    Y = []
    numBuckets = 12
    T_x = 412
    for i in range(0, len(df['item_description'])):#len(df['item_description'])):
        if (pd.isnull(df['item_description'][i]) == False): # Checks for Nan descriptions
            X.append((getIndexArrForSentence(df['item_description'][i])))
        else:
            X.append(np.zeros(T_x))
        Y.append(OneHot(int(df['price-bucket'][i]), numBuckets))
    Y= np.array(Y).T
    X = np.array(X).T
    return X, Y

def getIndexArrForSentence(sentence):
    T_x = 412
    arr = np.zeros(T_x)
    words = nltk.word_tokenize(sentence.lower())
    count = 0
    for w in words:
        if w.encode('UTF-8') in w2v:
            arr[count] = wordToIndex[w.encode('UTF-8')]
        count = count + 1
    return arr



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

def miniBatchIndicesToEmbedding(minibatch_X):
    m = minibatch_X.shape[1]
    T_x = 412 # Maximum number of time steps (zero-padded)
    n_x = 100 # Length of words2vec vector for each word
    newArr = np.zeros((m, T_x, n_x))
    for i in range(m): # Iterating through samples
        for j in range(T_x): # Iterating through words
            indexToW2v = 0
            indexToW2v = minibatch_X[j,i]
            if  indexToW2v != 0:
                newArr[i,j,:] = np.array(indexToEmb[indexToW2v])
    return newArr

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (number of examples, Tx)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = Y.shape[1]                 # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)    X shape: (Tx, m)   Y shape: (n_y, m)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))  # not sure why we need to reshape here

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

# ==========
#   MODEL
# ==========
def model(X_train, Y_train, X_dev, Y_dev, learning_rate = 0.01, num_epochs = 10000,
        batch_size = 128, display_step = 1):
    # Shape of X: (m, Tx, n_x)??? Emmie please check this
    # Shape of Y: (n_y, m)

    # Network Parameters
    Tx = 412 # Sequence max length
    n_hidden = 64 # hidden layer num of features
    n_y = 12 # linear sequence or not
    n_x = 100 # w2v length

    # trainset = ToySequenceData(n_samples=1000, max_seq_len=seq_max_len)
    # testset = ToySequenceData(n_samples=500, max_seq_len=seq_max_len)

    # tf Graph input
    X = tf.placeholder("float", [None, Tx, n_x])
    Y = tf.placeholder("float", [n_y, None])
    # A placeholder for indicating each sequence length
    #Tx = tf.placeholder(tf.int32, [None])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_y]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_y]))
    }

    pred = dynamicRNN(X, Tx, weights, biases, n_x)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = tf.transpose(pred), labels = Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(tf.transpose(pred)), tf.argmax(Y)) #Argmax over columns
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)
        for step in range(1, num_epochs + 1):
            # extract each miniminibatch_X, miniBatch_Y at each
            # minibatch_X, batch_y, seq_max_len = trainset.next(batch_size)
            # Run optimization op (backprop)
            #make minimatches here (randomly shuffling across m)
            minibatches = random_mini_batches(X_train, Y_train, mini_batch_size = 64, seed = 0)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                # Expand mininminibatch_X
                minibatch_X = miniBatchIndicesToEmbedding(minibatch_X)
                sess.run(optimizer, feed_dict={X: minibatch_X, Y: minibatch_Y})
                                               # Tx: Tx})
            if step % display_step == 0 or step == 1:
                # Calculate batch accuracy & loss
                acc, loss = sess.run([accuracy, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                                                    #Tx: Tx})
                print("Step " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

        print("Optimization Finished!")

        # # Calculate accuracy
        # test_data = testset.data
        # test_label = testset.labels
        # test_Tx = testset.Tx
        # print("Testing Accuracy:", \
        #     sess.run(accuracy, feed_dict={X: test_data, Y: test_label,
        #                                   Tx: test_Tx}))

    return

def dynamicRNN(X, Tx, weights, biases, n_x):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (m, Tx, n_x)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    X = tf.unstack(X, Tx, 1) #Unstack to be (None, 100) vectors
    n_hidden = 64
    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    Z_out, c = tf.contrib.rnn.static_rnn(lstm_cell, X, dtype=tf.float32)
                                #sequence_length=Tx)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'Z_out' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    Z_out = tf.stack(Z_out)
    Z_out = tf.transpose(Z_out, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(Z_out)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * Tx + (Tx - 1)
    # Indexing
    Z_out = tf.gather(tf.reshape(Z_out, [-1, n_hidden]), index)

    # Linear activation, using Z_out computed above
    return tf.matmul(Z_out, weights['out']) + biases['out']

if __name__ == '__main__':
    main()

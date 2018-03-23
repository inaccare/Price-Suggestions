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
import gensim
import multiprocessing


w2v = None
w2vFile = sys.argv[4]
w2v = gensim.models.word2vec.Word2Vec.load(w2vFile)

wordToIndex = dict()
indexToEmb = dict()
count = 1
for w in w2v.wv.vocab:
    wordToIndex[w] = count
    indexToEmb[count] = w2v.wv[w]
    count = count + 1

def main():

    # Usage is as follows: python model.py <train>.csv <dev>.csv <test>.csv <glove file>

    X_test = []
    Y_test = []
    testCSV = None
    trainCSV = None
    trainCSV = sys.argv[1]
    devCSV = sys.argv[2]
    testCSV = sys.argv[3]

    Tx = 72
    trainDF = pd.read_csv(trainCSV, header = 0)
    X_train, Y_train = getProductIndicesAndPrices(trainDF, Tx)
    devDF = pd.read_csv(devCSV, header = 0)
    X_dev, Y_dev = getProductIndicesAndPrices(devDF, Tx)
    testDF = pd.read_csv(testCSV, header = 0)
    X_test, Y_test = getProductIndicesAndPrices(testDF, Tx)

    print ("X_train shape: " + str(len(X_train)))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_dev shape: " + str(X_dev.shape))
    print ("Y_dev shape: " + str(Y_dev.shape))
    print ("X_dev shape: " + str(X_test.shape))
    print ("Y_dev shape: " + str(Y_test.shape))

    rebuildModel(X_train, X_dev, X_test, Y_train, Y_dev, Y_test)

def getProductIndicesAndPrices(df, T_x):
    X = []
    Y = []
    numBuckets = 12
    for i in range(0, len(df['item_description'])):#len(df['item_description'])):
        if (pd.isnull(df['item_description'][i]) == False): # Checks for Nan descriptions
            X.append( (getIndexArrForSentence(df['item_description'][i], T_x)) )
        else:
            X.append(np.zeros(T_x))
        Y.append(OneHot(int(df['price-bucket'][i]), numBuckets))
    Y= np.array(Y).T
    X = np.array(X).T
    return X, Y

def getIndexArrForSentence(sentence, T_x):
    arr = np.zeros(T_x)
    words = nltk.word_tokenize(sentence.lower())
    count = 0
    for w in words:
        # Only looking at first 72 words!
        if (count == T_x):
            break
        if w in w2v.wv.vocab:
            arr[count] = wordToIndex[w]
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
    T_x = 72
    n_x = 100 # Length of words2vec vector for each word
    newArr = np.zeros((m, T_x, n_x))
    for i in range(m): # Iterating through samples
        for j in range(T_x): # Iterating through words
            indexToW2v = 0
            indexToW2v = minibatch_X[j,i]
            if  indexToW2v != 0:
                newArr[i,j,:] = np.array(indexToEmb[indexToW2v])
    return newArr

def sequential_mini_batches(X, Y, mini_batch_size = 64):
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

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def rebuildModel(X_train, X_dev, X_test, Y_train, Y_dev, Y_test, Tx = 72, n_y = 12, n_x = 100):
    # tf Graph input
    X = tf.placeholder("float", [None, Tx, n_x])
    Y = tf.placeholder("float", [n_y, None])

    # Use the saver object normally after that.
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./model_onlydescriptions_deeplstm_w2v_expan_v2.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
# 
        graph = tf.get_default_graph()
        # [print("Name is " + n.name) for n in tf.get_default_graph().as_graph_def().node] # Did this to find recorded name for X and Y tensors

        # X = graph.get_tensor_by_name("X:0")
        X = graph.get_tensor_by_name("Placeholder_2:0") # Was supposed to be "X:0" but forgot to name tensor in model
        Y = graph.get_tensor_by_name("Placeholder_1_1:0") # Was supposed to be "Y:0" but forgot to name tensor in model


        dev_num_correct = miniBatch_calculation(X_train, Y_train, graph, X, Y, sess)
        print("Accuracy for train set: "+ str(dev_num_correct/X_train.shape[1]))

        dev_num_correct = miniBatch_calculation(X_dev, Y_dev, graph, X, Y, sess)
        print("Accuracy for dev set: "+ str(dev_num_correct/X_dev.shape[1]))

        test_num_correct = miniBatch_calculation(X_test, Y_test, graph, X, Y, sess)
        print("Accuracy for test set: "+ str(dev_num_correct/X_test.shape[1]))



def miniBatch_calculation(X_data, Y_data, graph, X, Y, sess, Tx = 72, mini_batch_size = 64, num_correct = 0):
    minibatches = sequential_mini_batches(X_data, Y_data, mini_batch_size = mini_batch_size)
    for minibatch in minibatches:
        (minibatch_X, minibatch_Y) = minibatch
        minibatch_X = miniBatchIndicesToEmbedding(minibatch_X)
        count_correct = graph.get_tensor_by_name("num_correct:0")
        num_correct_mb = sess.run(count_correct, feed_dict = {X:minibatch_X,Y:minibatch_Y})
        num_correct = num_correct + tf.cast(num_correct_mb, dtype=tf.int32)
    return sess.run(num_correct)


if __name__ == '__main__':
    main()

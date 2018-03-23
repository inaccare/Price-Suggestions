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
import multiprocessing
import gensim

"""
This model's architecture is as follows: product descriptions go into an LSTM layer, output of LSTM goes
through a 3-layer FC network. Other inputs (brand, categories and condition) get converted into one-hot
(in the case of brand and condition inputs) or multi-hot vectors (the case for categories). These other
inputs then get concatenated with output of forementioned 3-layer FC and get put through a 3-layer FC network.
Output is a softmax vector of length 12, corresponding to 12 distinct price buckets.
"""

num_categories_indices, num_brands_indices = 949, 4779
len_cat_vecs, len_brand_vecs = 950, 4780
len_cond_vecs = 6
w2v = None
w2vFile = sys.argv[3]
w2v = gensim.models.word2vec.Word2Vec.load(w2vFile)

wordToIndex = dict()
indexToEmb = dict()
count = 1
for w in w2v.wv.vocab:
    wordToIndex[w] = count
    indexToEmb[count] = w2v.wv[w]
    count = count + 1

def main():
    global num_categories_indices, num_brands_indices, brands_indices
    X_test = []
    Y_test = []
    testCSV = None
    trainCSV = None
    devCSV = None
    if (len(sys.argv) >=2):
        trainCSV = sys.argv[1]
    if (len(sys.argv) >= 4):
        devCSV = sys.argv[2]
    lr, ne, bs, tx = None, None, None, 72
    if (len(sys.argv) >= 5):
        lr, ne, bs, tx = getHyperparamsFromJSON(str(sys.argv[4]))
    if (len(sys.argv) >= 6):
        num_categories_indices, num_brands_indices = readCatsAndBrandsFromJSON(str(sys.argv[5]))
    len_cat_vecs = num_categories_indices + 1
    len_brand_vecs = num_brands_indices + 1
    trainDF = pd.read_csv(trainCSV, header = 0)
    # trainDF = trainDF.truncate(before = 0, after = 299999)
    trainCategories, trainBrands = getOtherArrs(trainDF)
    trainConditions = getConditions(trainDF)
    # For each entry in X_train, we have an array of length T_x with each entry
    # corresponding to an index into the word's w2v embedding
    X_train, Y_train = getProductIndicesAndPrices(trainDF, tx)
    devDF = pd.read_csv(devCSV, header = 0)
    devCategories, devBrands = getOtherArrs(devDF)
    devConditions = getConditions(devDF)
    X_dev, Y_dev = getProductIndicesAndPrices(devDF, tx)
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_dev shape: " + str(X_dev.shape))
    print ("Y_dev shape: " + str(Y_dev.shape))
    if (lr == None):
        model(X_train, Y_train, trainCategories, trainBrands, trainConditions, X_dev, Y_dev, devCategories, devBrands, devConditions)
    else:
        model(X_train, Y_train, trainCategories, trainBrands, trainConditions, X_dev, Y_dev, devCategories, devBrands, devConditions, learning_rate = lr, num_epochs = ne, mini_batch_size = bs, Tx = tx)

# ==========
#   MODEL
# ==========
def model(X_train, Y_train, trainCategories, trainBrands, trainConditions, X_dev, Y_dev, devCategories, devBrands, devConditions, learning_rate = 0.01, num_epochs = 10,
        mini_batch_size = 128, Tx = 72, display_step = 1, n_hidden = 64):
    # Shape of X: (m, Tx, n_x)??? Emmie please check this
    # Shape of Y: (n_y, m)
    print ("Model has following hyperparameters: learning rate: " + str(learning_rate) + ", num_epochs: " + str(num_epochs) + ", mini_batch_size: " \
        + str(mini_batch_size) + ", Tx: "+ str(Tx) + ".")

    # hidden layer num of features
    n_y = 12 # linear sequence or not
    n_x = 100 # w2v length

    # tf Graph input
    X = tf.placeholder("float", [None, Tx, n_x])
    Y = tf.placeholder("float", [n_y, None])
    Cats = tf.placeholder("int32", [None, len_cat_vecs])
    Brands = tf.placeholder("int32", [None, len_brand_vecs])
    Conds = tf.placeholder("int32", [None,len_cond_vecs])
    # A placeholder for indicating each sequence length
    #Tx = tf.placeholder(tf.int32, [None])

    # Define weights
    weights = {
        # 'out': tf.Variable(tf.random_normal([n_hidden, n_y]))
        'W_1' : tf.get_variable('W_1',[n_hidden,n_hidden], initializer = tf.contrib.layers.xavier_initializer(seed = 1)),
        'W_2' : tf.get_variable('W_2',[n_hidden,n_hidden], initializer = tf.contrib.layers.xavier_initializer(seed = 1)),
        'W_out' : tf.get_variable('W_out',[n_hidden, n_hidden], initializer = tf.contrib.layers.xavier_initializer(seed = 1)),
        'W_f1' : tf.get_variable('W_f1',[n_hidden + len_cat_vecs + len_brand_vecs + len_cond_vecs ,n_hidden], initializer = tf.contrib.layers.xavier_initializer(seed = 1)),
        'W_f2' : tf.get_variable('W_f2',[n_hidden,n_hidden], initializer = tf.contrib.layers.xavier_initializer(seed = 1)),
        'W_fout' : tf.get_variable('W_fout',[n_hidden,n_y], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    }
    biases = {
        # 'out': tf.Variable(tf.random_normal([n_y]))
        'b_1' : tf.get_variable('b_1',[n_hidden], initializer = tf.zeros_initializer()),
        'b_2' : tf.get_variable('b_2',[n_hidden], initializer = tf.zeros_initializer()),
        'b_out' : tf.get_variable('b_out',[n_hidden], initializer = tf.zeros_initializer()),
        'b_f1' : tf.get_variable('b_f1',[n_hidden], initializer = tf.zeros_initializer()),
        'b_f2' : tf.get_variable('b_f2',[n_hidden], initializer = tf.zeros_initializer()),
        'b_fout' : tf.get_variable('b_fout',[n_y], initializer = tf.zeros_initializer())
    }

    pred = dynamicRNN(X,Cats, Brands,Conds, Tx, weights, biases, n_x, n_hidden)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = tf.transpose(Y)))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(tf.transpose(Y), 1)) #Argmax over columns
    num_correct = tf.reduce_sum(tf.cast(correct_pred, tf.float32), name = "num_correct")

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Initialize the saver
    saver = tf.train.Saver()

    m = Y_train.shape[1]
    num_minibatches = int(math.floor(m/mini_batch_size))
    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        for step in range(1, num_epochs + 1):
            epoch_cost =0
            tot_num_correct = 0
            # extract each miniminibatch_X, miniBatch_Y at each
            #make minimatches here (randomly shuffling across m)
            minibatches = random_mini_batches(X_train, Y_train, trainCategories, trainBrands,trainConditions, mini_batch_size = mini_batch_size, seed = 0)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y, mini_batch_categories, mini_batch_brands, mini_batch_conds) = minibatch
                # Expand mininminibatch_X 
                minibatch_X = miniBatchIndicesToEmbedding(minibatch_X, Tx)
                mini_batch_categories, mini_batch_brands, mini_batch_conds = expandOtherArrs(mini_batch_categories, mini_batch_brands, len_cat_vecs, len_brand_vecs, mini_batch_conds)
                # print ("Shape of minibatch_X is " + str(minibatch_X.shape))
                sess.run(optimizer, feed_dict={X: minibatch_X, Y: minibatch_Y, Cats: mini_batch_categories, Brands: mini_batch_brands, Conds: mini_batch_conds})
                mini_num_correct, loss = sess.run([num_correct, cost], feed_dict={X: minibatch_X, Y: minibatch_Y, Cats: mini_batch_categories, Brands: mini_batch_brands, Conds: mini_batch_conds})
                epoch_cost = epoch_cost + loss
                tot_num_correct = tot_num_correct + mini_num_correct
                                               # Tx: Tx})
            if step % display_step == 0 or step == 1:
                # Calculate batch accuracy & loss
                                                    #Tx: Tx})
                print("Epoch " + str(step) + ", Cost= " + \
                      "{:.6f}".format(epoch_cost/num_minibatches) + ", Training Accuracy= " + \
                      "{:.5f}".format(float(tot_num_correct/m)))

        print("Optimization Finished!")
        train_num_correct = 0
        minibatches = random_mini_batches(X_train, Y_train, trainCategories, trainBrands, trainConditions,mini_batch_size = mini_batch_size, seed = 0)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y, mini_batch_categories, mini_batch_brands, mini_batch_conds) = minibatch
            minibatch_X = miniBatchIndicesToEmbedding(minibatch_X, Tx)
            mini_batch_categories, mini_batch_brands, mini_batch_conds = expandOtherArrs(mini_batch_categories, mini_batch_brands, len_cat_vecs, len_brand_vecs, mini_batch_conds)
            num_correct_mb, loss = sess.run([num_correct, cost], feed_dict={X: minibatch_X, Y: minibatch_Y, Cats: mini_batch_categories, Brands: mini_batch_brands, Conds: mini_batch_conds})
            train_num_correct = train_num_correct + num_correct_mb
        print("Accuracy for train set: "+ str(train_num_correct/X_train.shape[1]))

        dev_num_correct = 0
        minibatches = random_mini_batches(X_dev, Y_dev, devCategories, devBrands, devConditions,mini_batch_size = mini_batch_size, seed = 0)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y, mini_batch_categories, mini_batch_brands, mini_batch_conds) = minibatch
            minibatch_X = miniBatchIndicesToEmbedding(minibatch_X, Tx)
            mini_batch_categories, mini_batch_brands, mini_batch_conds = expandOtherArrs(mini_batch_categories, mini_batch_brands, len_cat_vecs, len_brand_vecs, mini_batch_conds)
            num_correct_mb, loss = sess.run([num_correct, cost], feed_dict={X: minibatch_X, Y: minibatch_Y, Cats: mini_batch_categories, Brands: mini_batch_brands, Conds: mini_batch_conds})
            dev_num_correct = dev_num_correct + num_correct_mb
        print("Accuracy for dev set: "+ str(dev_num_correct/X_dev.shape[1]))
        saver.save(sess, './model_lstm_w2v_expan_v2')
        sess.close()
        # # Calculate accuracy
        # test_data = testset.data
        # test_label = testset.labels
        # test_Tx = testset.Tx
        # print("Testing Accuracy:", \
        #     sess.run(accuracy, feed_dict={X: test_data, Y: test_label,
        #                                   Tx: test_Tx}))
    return

def dynamicRNN(X, Cats, Brands, Conds, Tx, weights, biases, n_x, n_hidden):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (m, Tx, n_x)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input); or Tx tensors of shape (m, n_x)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    X = tf.unstack(X, Tx, 1) #Unstack to be (None, 100) vectors
    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    # Z_out, c = tf.contrib.rnn.static_rnn(lstm_cell, X, dtype=tf.float32)
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
    # Deepen LSTM network with fully connected
    Z_out = tf.matmul(Z_out, weights['W_1']) + biases['b_1']
    Z_out = tf.nn.relu(Z_out)
    dropout = tf.layers.dropout(inputs=Z_out, rate=0.1) # Dropout 10% of units
    Z_out = tf.matmul(Z_out, weights['W_2']) + biases['b_2']
    Z_out = tf.nn.relu(Z_out)
    dropout = tf.layers.dropout(inputs=Z_out, rate=0.1)
    Z_out = tf.matmul(dropout, weights['W_out']) + biases['b_out']
    Z_out = tf.nn.relu(Z_out)     

    Others = tf.concat([Cats, Brands,Conds ], axis = 1)

    Z_out = tf.cast(Z_out, tf.int32)

    Z_out = tf.concat([Z_out, Others], axis = 1)

    Z_out = tf.cast(Z_out, tf.float32)

    # Deepen Full Netowrk
    Z_out = tf.matmul(Z_out, weights['W_f1']) + biases['b_f1']
    Z_out = tf.nn.relu(Z_out)
    Z_out = tf.layers.dropout(inputs=Z_out, rate=0.1) # Dropout 10% of units
    Z_out = tf.matmul(Z_out, weights['W_f2']) + biases['b_f2']
    Z_out = tf.nn.relu(Z_out)
    Z_out = tf.layers.dropout(inputs=Z_out, rate=0.1)
    Z_out = tf.matmul(Z_out, weights['W_fout']) + biases['b_fout']

    # Linear activation, using Z_out computed above
    return Z_out


# =================================================================
#   Helper functions for reading in data/getting minibatches ready
# =================================================================
def getHyperparamsFromJSON(filename):
    parameters = None
    with open(filename, 'r') as fp:
        parameters = json.load(fp)
    return float(parameters['learning_rate']), int(parameters['num_epochs']), int(parameters['batch_size']), int(parameters['Tx'])

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
    Creates onehot vector for our Y output
    Arguments:
    bucket -- index of correct bucket for example in Y
    numBuckets -- number of buckets used to split of the prices of objects
    Returns:
    arr -- onehot array
    """
    arr = np.zeros(numBuckets)
    arr[bucket] = 1
    return arr

def miniBatchIndicesToEmbedding(minibatch_X, T_x):
    m = minibatch_X.shape[1]# Maximum number of time steps (zero-padded)
    n_x = 100 # Length of words2vec vector for each word
    newArr = np.zeros((m, T_x, n_x))
    for i in range(m): # Iterating through samples
        for j in range(T_x): # Iterating through words
            indexToW2v = 0
            indexToW2v = minibatch_X[j,i]
            if  indexToW2v != 0:
                newArr[i,j,:] = np.array(indexToEmb[indexToW2v])
    return newArr

def random_mini_batches(X, Y, categories, brands, conds, mini_batch_size = 64, seed = 0):
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
    print("shape X", X.shape)
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))  # not sure why we need to reshape here
    shuffled_categories = categories[permutation]
    shuffled_brands = brands[permutation]
    shuffled_conds = conds[permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_categories = shuffled_categories[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_brands = shuffled_brands[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_conds = shuffled_conds[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_categories, mini_batch_brands, mini_batch_conds)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_categories = shuffled_categories[num_complete_minibatches * mini_batch_size : m]
        mini_batch_brands = shuffled_brands[num_complete_minibatches * mini_batch_size : m]
        mini_batch_conds = shuffled_conds[num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_categories, mini_batch_brands, mini_batch_conds)
        mini_batches.append(mini_batch)

    return mini_batches

def getConditions(df):
    conditions = []
    for i in range(len(df['price'])):
        if (pd.isnull(df['item_condition_id'][i]) == False):
            conditions.append(int(df['item_condition_id'][i]) - 1) # -1 so that it is the index
        else:
            conditions.append(int(len_cond_vecs) - 1)
    return np.array(conditions)

def getOtherArrs(df):
    categories = []
    brands = []
    for i in range(len(df['price'])):
        brands.append(int(df['brand-indices'][i]))
        categories.append(df['categories-indices'][i])
    return np.array(categories), np.array(brands)

def readCatsAndBrandsFromJSON(filename):
    Vars = None
    with open(filename, 'r') as fp:
        Vars = json.load(fp)
    return int(Vars['num_categories_indices']), int(Vars['num_brands_indices'])

def expandOtherArrs(categories_conden, brands_conden, len_cat_vecs, len_brand_vecs, conds):
    categories = []
    brands = []
    conditions = []
    for i in range(brands_conden.shape[0]):
        categories.append(MultiHot(categories_conden[i], len_cat_vecs))
        brands.append(OneHot(brands_conden[i], len_brand_vecs))
        conditions.append(OneHot(conds[i], len_cond_vecs))
    return np.array(categories), np.array(brands), np.array(conditions)

def MultiHot(List, numBuckets):
    arr = np.zeros(numBuckets)
    # Have to do it like this because read_csv defaults the List to really be a string that visibly looks like a list (eg. "[0 1 3]")
    for i in List.split(' '):
        if i.isdigit():
            arr[(int)(i)] = 1
    return arr    

if __name__ == '__main__':
    main()

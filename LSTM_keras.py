import numpy as np
import sys
import nltk
nltk.download('punkt')
import pandas as pd

from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras import backend as K
np.random.seed(1)

w2v = None
with open("glove.6B.100d.txt", "rb") as lines:
    w2v = {line.split()[0]: np.array(line.split()[1:])
            for line in lines}

def main():

    X_test = []
    Y_test = []
    testCSV = None
    trainCSV = None
    if (len(sys.argv) >=2):
        trainCSV = sys.argv[1]
    if (len(sys.argv) == 3):
        devCSV = sys.argv[2]
    trainDF = pd.read_csv(trainCSV, header = 0)
    X_train, Y_train = getProductSentencesAndPrices(trainDF)
    devDF = pd.read_csv(devCSV, header = 0)
    X_dev, Y_dev = getProductSentencesAndPrices(devDF)
    print ("X_train shape: " + str(len(X_train)))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_dev shape: " + str(X_dev.shape))
    print ("Y_dev shape: " + str(Y_dev.shape))

    Tx = 80;
    maxLen = 412

    model = Emojify_V2((maxLen,), Tx, maxLen)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    cost = model.fit(X_train, Y_train, epochs = 1, batch_size = 32, shuffle=True)
    print(cost)


def OneHot(bucket, numBuckets):
    arr = np.zeros(numBuckets)
    arr[bucket] = 1
    return arr

def getProductSentencesAndPrices(df):
    X = []
    Y = []
    numBuckets = 12
    for i in range(0, 10):#len(df['item_description'])):
        X.append(nltk.word_tokenize(str(df['item_description'][i])))
        Y.append(OneHot(int(df['price-bucket'][i]), numBuckets))
    Y= np.array(Y).T
    X = np.array(X).T
    return X, Y

# Stuck on for loops over tensors

def wordsToEmebddings(minibatch_X, Tx, maxLength):
    """
    Expands the minibatch sentences into word2vec vectors for each word
    Arguments:
    minibatch_X -- training examples of the current minibatch
    Returns:
    mini_batch_vectors -- training expamples as binary vectors
    """
    X = np.zeros((len(K.transpose(minibatch_X)), maxLength, Tx))  # shape (m, n_x, Tx) where n_x is length of input vectors and Tx is timesteps
    for i in m:
        for j in maxLength:
            
            temp = w2v[minibatch_X[i][j]]
            X.append(sample, temp, word)
    return X

def zeroPadArrays(minibatch_X, maxLength):
    """
    Zero pads the minibatch descriptions to the maximum length description
    Arguments:
    minibatch_X -- training examples of the current minibatch
    maxLength -- Length of longest description
    Returns:
    padded_batch -- minibatch with padded descriptions
    """
    padded_batch = []
    map_fn(lambda description: x * x, elems)
    for description in K.transpose(minibatch_X):   # Loop over the rows
        str_len = len(description)
        padded_str = np.pad(description, (0, maxLength - str_len), 'constant', constant_values=(0))
        padded_batch.append(padded_str)
    return padded_batch.T     # Each column is one training example

def sentences_to_embeddings(X, max_len, Tx):  # Sentences to embedding
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4).

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    m = X.shape[0]                                   # number of training examples

    z = zeroPadArrays(X, max_len)
    X_emb = wordsToEmebddings(z, Tx, max_len)


    return X_emb


def Emojify_V2(input_shape, Tx, max_len):
    """
    Function creating the Product Description model's graph.

    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """

    ### START CODE HERE ###
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentences = Input(shape = input_shape, dtype = 'int32')

    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = sentences_to_embeddings(sentences, Tx, max_len)

    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_embeddings)

    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = LSTM(128, return_sequences = True)(embeddings)    # This is 3D...
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(5)(X)
    # Add a softmax activation
    X = Activation('softmax')(X)

    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs = [sentences], outputs = [X])

    ### END CODE HERE ###

    return model

if __name__ == '__main__':
    main()

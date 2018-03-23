# Price-Suggestions

The data that we are using can be obtained at (https://www.kaggle.com/c/mercari-price-suggestion-challenge).  The files provided at that link contain over 1.4 million items along with their prices, descriptions, conditions, etc.

makeCSVs.py is used to split the data into train, dev, and test sets from the train dataset in the Kaggle challenge. Did not use their and test set because they did not have labeled prices.

build_na_boW_conden.py allows us to put the data into a more useful form to feed into our neural network.  It currently grabs only the item description for each item and represents it as a bag of words.  Specifically, it outputs new files *_enc.csv* in which the last column contains a vector of indices corresponding to which index in the vocabulary corresponds to each word in the item description.

model_1_nn_boW.py takes in these *_enc.csv* files and expands the vectors into vectors the size of the vocabulary with a 1 or 0 corresponding to whether or not each word appears in the item description.  This is then fed into a 2 layer NN with a softmax activation function corresponding to which of the 12 buckets the item's price is expected to be in

build_na_w2v_avg.py uses glove.6B.100d.txt and for each item description, averages the word2vec vectors corresponding to each word in the item description.  It then outputs these averaged vectors into an *_enc.csv* file next to the item description each vector pertains to

model_2_w2vGlove.py uses these averaged word2vec vectors as inputs into a 2 layer NN with softmax activation function

build_w2v_vector.py takes in the corpus from all of our product descriptions and builds w2v encodings.

model_3_lstmw2vglove.py has an architecture as follows: product descriptions goes into an LSTM cell where each
word's w2v encoding is used as an input for each time step of the lstm. This is done for the first
T_x words of each description. Output of last lstm timestep then goes through a softmax output layer
and output is softmax vector of length 12 corresponding to 12 distinct price buckets. Pre-trained 
glove-based w2v encodings used.

model_4_lstmw2vtrained.py is the same as model_3_lstmw2vglove.py except that it uses the encodings computed in model_2_w2vGlove.py.

model_5_lstmandothers12.py has an architecture defined as follows: product descriptions goes into an LSTM cell where each
word's w2v encoding is used as an input for each time step of the lstm. This is done for the first
T_x words of each description. Output of last lstm timestep then gets concatenated with encodings of
other inputs (one-hot brand vector, one-hot condition vector and multi-hot categories) before being fed 
into a softmax output layer. Output is softmax vector of length 12, for 12 disttinct price buckets.
W2v encodings trained on our corpus is used.

model_6_lstmandothers.py is essentially the same as model_5_lstmandothers12.py except that it uses a softmax vector of length 20 instead of 12 since it uses 20 different buckets instead of 20.

model_7_lstmandothers.py is essentially the same as model_5_lstmandothers12.py except that the output layer is a linear function rather than a softmax output. Output for each sample is a scalar value corresponding to a predicted price. Cost function used for all linear output models is Root Mean Squared Logarithmic Error.

model_8_deeplstmandothers.py has an architecture defined as follows: product descriptions go into an LSTM layer, output of LSTM goes
through a 3-layer FC network. Other inputs (brand, categories and condition) get converted into one-hot
(in the case of brand and condition inputs) or multi-hot vectors (the case for categories). These other
inputs then get concatenated with output of forementioned 3-layer FC and get put through a 3-layer FC network.
Output is a softmax vector of length 20, corresponding to 20 distinct price buckets.

model_9_deeplstmandotherslinear.py has an architecture identical to model_8_deeplstmandothers.py with the exception of the output layer. In this model, we are again using a linear output function. The output for each sample is a scalar value corresponding to the predicted price. Cost function used for all linear output models is Root Mean Squared Logarithmic Error.

model_10_deeplstmandothers12.py has an architecture identical to model_8_deeplstmandothers.py, except that the output softmax vector is of length 12, corresponding to 12 distinct price buckets.

model_11_deeplstmonlydes12.py has an architecture defined as follows: product descriptions go into an LSTM layer, output of LSTM goes
through a 6-layer FC network. Output is a softmax vector of length 12, corresponding to 12 distinct 
price buckets.

model_12_deeplstmonlydes.py has an architecture identical to model_11_deeplstmonlydes12.py except that the output softmax vector is of length 20 instead of 12, corresponding to the 20 distinct price buckets used for this classification task.

model_13_deeplstmonlydeslinear.py has an architecture identical to model_11_deeplstmonlydes12.py except that instead of a softmax output, we have a linear output.


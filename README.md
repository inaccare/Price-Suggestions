# Price-Suggestions

The data that we are using can be obtained at (https://www.kaggle.com/c/mercari-price-suggestion-challenge).  The files provided at that link contain over 1.4 million items along with their prices, descriptions, conditions, etc.

makeCSVs.py is used to split the data into train, dev, and test sets

bag_of_words.py allows us to put the data into a more useful form to feed into our neural network.  It currently grabs only the item description for each item and represents it as a bag of words.  Specifically, it outputs new files *_enc.csv* in which the last column contains a vector of indices corresponding to which index in the vocabulary corresponds to each word in the item description.

model.py takes in these *_enc.csv* files and expands the vectors into vectors the size of the vocabulary with a 1 or 0 corresponding to whether or not each word appears in the item description.  This is then fed into a 2 layer NN with a softmax activation function corresponding to which of the 12 buckets the item's price is expected to be in

word2vec.py uses glove.6B.100d.txt and for each item description, averages the word2vec vectors corresponding to each word in the item description.  It then outputs these averaged vectors into an *_enc.csv* file next to the item description each vector pertains to

model_word2vec.py uses these averaged word2vec vectors as inputs into a 2 layer NN with softmax activation function

model.py takes in the output of bag_of_words.py and runs it through a simple neural network and outputs a softmax vector which indicates the likelihood of the item being in each of the price buckets we defined.

Next Steps:
- Change output to being a linear function
- Tune parameters
- Begin using an RNN framework with LSTM
